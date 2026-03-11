"""RL algorithms and neural policy models for the market-impact MDP.

This module contains:

* **Tabular methods** -- ``sarsa`` (on-policy TD) and ``q_learning``
  (off-policy TD), each running a single episode and updating a Q-table
  in-place.
* **Policy-gradient method** -- ``reinforce`` trains a neural-network policy
  using the REINFORCE algorithm with baseline normalization and gradient
  clipping.  ``reinforce_with_options`` is an ablation-ready variant that
  lets the caller toggle baseline and clipping independently.
* **Neural-network policy** -- ``MLPPolicyNet`` is a flexible MLP whose
  architecture is selected via ``build_policy_net``.
* **Helpers** -- ``make_input`` normalizes a discrete state into a float
  tensor, ``generate_trajectories_batch`` rolls out K trajectories in
  parallel on the GPU/MPS device, ``set_all_seeds`` ensures reproducibility.

All functions receive MDP parameters through a ``config`` dict (see
main.ipynb Section 2).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .environment import epsilon_greedy, transition


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _start_state(config: Dict[str, object]) -> List[int]:
    """Return the MDP initial state [X_0, S_0, t=0].

    Inventory starts at NUM_BLOCKS - 1 (zero-indexed), price at S0, time at 0.
    """
    return [int(config["NUM_BLOCKS"]) - 1, int(config["S0"]), 0]


def _device(config: Dict[str, object]) -> torch.device:
    """Extract the torch device from config (cpu / mps / cuda)."""
    return config["DEVICE"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Neural-network policy
# ---------------------------------------------------------------------------

class MLPPolicyNet(nn.Module):
    """Multi-layer perceptron that maps a normalized state to action logits.

    The network takes a 3-dimensional input (t_norm, x_norm, s_norm) and
    outputs raw logits over ``num_actions`` actions.  Invalid actions (selling
    more than current inventory) are masked *outside* this class before
    applying softmax.

    Parameters
    ----------
    hidden_layers : sequence of int
        Width of each hidden layer.  E.g. [64, 64] gives two hidden layers
        of 64 units each, both followed by ReLU.
    num_actions : int
        Size of the action space (4 in the default MDP).

    Used in
    -------
    Instantiated by ``build_policy_net``; trained by ``reinforce`` and
    ``reinforce_with_options``.  The trained net is later used in
    ``greedy_action_from_net`` (analysis.py) for policy evaluation
    (main.ipynb Section 9).
    """

    def __init__(self, hidden_layers: Sequence[int], num_actions: int):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 3  # input: (t_norm, x_norm, s_norm)
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_policy_net(
    arch_name: str,
    num_actions: int,
    device: torch.device,
) -> MLPPolicyNet:
    """Factory: create an ``MLPPolicyNet`` by architecture name.

    Available architectures (see main.ipynb Section 5):

    ========  ===============  ~~params
    tiny      [8]                  ~36
    shallow   [32]                ~164
    medium    [64, 64]          ~4 612
    deep      [128, 128, 128]  ~33 668
    ========  ===============  ~~params

    Returns
    -------
    MLPPolicyNet already moved to ``device``.
    """
    arch = arch_name.lower()
    if arch == "tiny":
        hidden_layers = [8]
    elif arch == "shallow":
        hidden_layers = [32]
    elif arch == "medium":
        hidden_layers = [64, 64]
    elif arch == "deep":
        hidden_layers = [128, 128, 128]
    else:
        raise ValueError("arch_name must be one of: 'tiny', 'shallow', 'medium', 'deep'")
    return MLPPolicyNet(hidden_layers=hidden_layers, num_actions=num_actions).to(device)


def count_trainable_params(model: nn.Module) -> int:
    """Return the total number of trainable parameters in a PyTorch model.

    Used in main.ipynb Section 5 to print a table of architecture sizes.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# State normalization
# ---------------------------------------------------------------------------

def make_input(t: int, x: int, s: int, config: Dict[str, object]) -> torch.Tensor:
    """Normalize a discrete (t, x, s) state into a [0, 1]-ranged float tensor.

    Each component is linearly scaled so the network sees values in a
    comparable range regardless of the grid dimensions:

    * t_norm = t / (T - 1)
    * x_norm = x / (N - 1)
    * s_norm = s / (M - 1)

    Returns a 1-D tensor of shape (3,) on the configured device.

    Used in
    -------
    ``greedy_action_from_net`` (analysis.py) calls this to build a single-state
    input for the trained policy network during evaluation.
    ``generate_trajectories_batch`` mirrors the same normalization in a
    vectorized fashion.
    """
    num_time_steps = int(config["NUM_TIME_STEPS"])
    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])

    t_norm = t / (num_time_steps - 1) if num_time_steps > 1 else 0.0
    x_norm = x / (num_blocks - 1) if num_blocks > 1 else 0.0
    s_norm = s / (num_s - 1) if num_s > 1 else 0.0

    return torch.tensor([t_norm, x_norm, s_norm], dtype=torch.float32, device=_device(config))


# ---------------------------------------------------------------------------
# Batched trajectory generation (used by REINFORCE)
# ---------------------------------------------------------------------------

def generate_trajectories_batch(
    net: nn.Module,
    batch_size: int,
    config: Dict[str, object],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Roll out ``batch_size`` trajectories in parallel under policy ``net``.

    This is the inner loop of REINFORCE: for each of the K trajectories we
    sample actions from the policy, record log-probabilities, and accumulate
    rewards -- all in a single batched pass on the GPU/MPS device.

    Parameters
    ----------
    net : MLPPolicyNet
        Current policy network (not updated here).
    batch_size : int
        Number of trajectories to roll out simultaneously (= K in the
        REINFORCE notation).
    config : dict
        Full MDP parameter dictionary.

    Returns
    -------
    log_probs_per_step : list of Tensor, length T-1
        Each element has shape (batch_size,) and contains
        log pi(a_t | s_t) for the sampled action.
    total_r : Tensor of shape (batch_size,)
        Cumulative return for each trajectory.

    Used in
    -------
    Called by ``reinforce``, ``reinforce_with_options`` (this module) and
    ``plot_reinforce_loss_landscape`` (analysis.py).
    """
    device = _device(config)
    actions = list(config["ACTIONS"])
    actions_tensor = torch.tensor(actions, device=device)

    num_time_steps = int(config["NUM_TIME_STEPS"])
    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])
    dt = float(config["dt"])
    sigma = float(config["sigma"])
    nu = float(config["nu"])
    mu = float(config["mu"])
    lmbda = float(config["lmbda"])
    block_size = int(config["BLOCK_SIZE"])

    sqrt_dt = float(np.sqrt(dt))
    # Pre-compute constants for the log-normal variance formula:
    #   Var[S] = S^2 * exp(2*mu*dt + sigma^2*dt) * (exp(sigma^2*dt) - 1)
    exp_reward_a = float(np.exp(2 * mu * dt + sigma**2 * dt))
    exp_reward_b = float(np.exp(sigma**2 * dt) - 1.0)

    start = _start_state(config)
    x = torch.full((batch_size,), int(start[0]), dtype=torch.long, device=device)
    s = torch.full((batch_size,), int(start[1]), dtype=torch.long, device=device)

    log_probs_per_step: List[torch.Tensor] = []
    total_r = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for t in range(num_time_steps - 1):
        # Normalize state components to [0, 1] (same logic as make_input)
        t_norm = t / (num_time_steps - 1) if num_time_steps > 1 else 0.0
        x_norm = x.float() / (num_blocks - 1) if num_blocks > 1 else torch.zeros_like(x, dtype=torch.float32)
        s_norm = s.float() / (num_s - 1) if num_s > 1 else torch.zeros_like(s, dtype=torch.float32)

        inp = torch.stack(
            [
                torch.full((batch_size,), float(t_norm), device=device),
                x_norm,
                s_norm,
            ],
            dim=1,
        )

        logits = net(inp)
        # Mask invalid actions: cannot sell more blocks than currently held.
        # For each trajectory, actions_tensor[a] <= x[k] is True iff action a
        # is feasible.  Infeasible logits are set to -1e9 so softmax assigns
        # them ~zero probability.
        valid_mask = actions_tensor.unsqueeze(0) <= x.unsqueeze(1)
        masked_logits = logits.masked_fill(~valid_mask, -1e9)
        probs = torch.softmax(masked_logits, dim=1)

        # Sample one action per trajectory and record its log-probability
        sampled_action_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        chosen_prob = probs.gather(1, sampled_action_idx.unsqueeze(1)).squeeze(1)
        log_probs_per_step.append(torch.log(chosen_prob + 1e-8))

        action = actions_tensor[sampled_action_idx]
        action = torch.minimum(action, x)  # safety clip
        x_next = x - action

        # Price dynamics (vectorized version of environment.transition)
        s_float = s.float()
        s_next = s_float * torch.exp(1.0 - nu * action.float()) + sigma * s_float * sqrt_dt * torch.randn(
            batch_size,
            device=device,
        )
        s_next = torch.ceil(s_next).clamp(0, num_s - 1).long()

        # Reward = execution revenue - inventory risk penalty
        var = (s_next.float() ** 2) * exp_reward_a * exp_reward_b
        reward = block_size * action.float() * s_float - lmbda * block_size * (x_next.float() ** 2) * var

        total_r += reward
        x, s = x_next, s_next

    return log_probs_per_step, total_r


# ---------------------------------------------------------------------------
# Tabular algorithms
# ---------------------------------------------------------------------------

def q_learning(
    q_table: np.ndarray,
    config: Dict[str, object],
    eta: float | None = None,
    eps: float | None = None,
) -> float:
    """Run one episode of Q-learning and update ``q_table`` in-place.

    Q-learning is **off-policy**: the TD target uses the greedy next-action
    value max_a' Q(s', a') regardless of the action actually taken.

    Parameters
    ----------
    q_table : ndarray of shape (NUM_BLOCKS, NUM_S, NUM_TIME_STEPS, |A|)
        Updated in-place after each transition.
    config : dict
    eta : float, optional
        Learning rate (defaults to config["eta"]).
    eps : float, optional
        Exploration rate (defaults to config["epsilon"]).

    Returns
    -------
    episode_return : float
        Sum of rewards collected during the episode.

    Used in
    -------
    Called in tight loops inside ``compare_sarsa_qlearning``,
    ``compare_all_architectures``, ``compare_fixed_trajectories_with_tables``,
    ``tabular_lr_sensitivity`` (analysis.py).
    """
    if eta is None:
        eta = float(config["eta"])
    if eps is None:
        eps = float(config["epsilon"])

    num_time_steps = int(config["NUM_TIME_STEPS"])
    current_state = _start_state(config)
    rewards = 0.0

    while current_state[2] < num_time_steps - 1:
        selected_action = epsilon_greedy(current_state, q_table, config, eps)
        next_state, reward = transition(current_state, selected_action, config)
        rewards += reward

        q_current = q_table[current_state[0], current_state[1], current_state[2], selected_action]
        # Off-policy target: use the *best* next-action value (greedy)
        q_next_max = np.max(q_table[next_state[0], next_state[1], next_state[2], :])
        q_table[current_state[0], current_state[1], current_state[2], selected_action] += eta * (
            reward + q_next_max - q_current
        )
        current_state = next_state

    return float(rewards)


def sarsa(
    q_table: np.ndarray,
    config: Dict[str, object],
    eta: float | None = None,
    eps: float | None = None,
) -> float:
    """Run one episode of SARSA and update ``q_table`` in-place.

    SARSA is **on-policy**: the TD target uses the value of the action
    actually selected by the epsilon-greedy policy at the next state,
    Q(s', a'), rather than the greedy max.

    Parameters
    ----------
    q_table : ndarray of shape (NUM_BLOCKS, NUM_S, NUM_TIME_STEPS, |A|)
        Updated in-place.
    config : dict
    eta : float, optional
        Learning rate.
    eps : float, optional
        Exploration rate.

    Returns
    -------
    episode_return : float

    Used in
    -------
    Same call-sites as ``q_learning`` above.
    """
    if eta is None:
        eta = float(config["eta"])
    if eps is None:
        eps = float(config["epsilon"])

    num_time_steps = int(config["NUM_TIME_STEPS"])
    current_state = _start_state(config)
    current_action = epsilon_greedy(current_state, q_table, config, eps)
    episode_return = 0.0

    while current_state[2] < num_time_steps - 1:
        next_state, reward = transition(current_state, current_action, config)
        episode_return += reward
        next_action = epsilon_greedy(next_state, q_table, config, eps)

        q_current = q_table[current_state[0], current_state[1], current_state[2], current_action]
        # On-policy target: use the action that *will* be taken (next_action)
        q_next = q_table[next_state[0], next_state[1], next_state[2], next_action]
        q_table[current_state[0], current_state[1], current_state[2], current_action] += eta * (
            reward + q_next - q_current
        )
        current_state = next_state
        current_action = next_action

    return float(episode_return)


# ---------------------------------------------------------------------------
# REINFORCE (policy gradient)
# ---------------------------------------------------------------------------

def reinforce(
    config: Dict[str, object],
    num_episodes: int = 1000,
    K: int = 50,
    lr: float = 3e-4,
    arch: str = "medium",
    clip_grad: float = 1.0,
) -> Tuple[nn.Module, List[float]]:
    """Train a neural-network policy with REINFORCE (baseline + clipping ON).

    Each "episode" (= one parameter update) rolls out K trajectories in
    parallel, computes the policy-gradient loss with return normalization,
    and updates the network via Adam.

    Parameters
    ----------
    config : dict
    num_episodes : int
        Number of gradient updates.
    K : int
        Trajectories per update (batch size).
    lr : float
        Adam learning rate.
    arch : str
        Architecture name passed to ``build_policy_net``.
    clip_grad : float
        Max global gradient norm for clipping.

    Returns
    -------
    net : MLPPolicyNet
        The trained policy network.
    returns : list of float, length num_episodes
        Average trajectory return at each update step.

    Used in
    -------
    Called by ``compare_all``, ``compare_all_architectures``,
    ``compare_fixed_trajectories_with_tables``,
    ``compare_fixed_updates_with_tables`` (analysis.py).
    Indirectly used in main.ipynb Sections 7, 8.
    """
    device = _device(config)
    actions = list(config["ACTIONS"])

    net = build_policy_net(arch_name=arch, num_actions=len(actions), device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    returns: List[float] = []

    for episode in tqdm(range(num_episodes), desc=f"REINFORCE-{arch}"):
        optimizer.zero_grad()
        log_probs_per_step, batch_returns_tensor = generate_trajectories_batch(net, K, config)

        avg_r = float(batch_returns_tensor.mean().item())
        returns.append(avg_r)

        # Baseline normalization: subtract mean, divide by std.
        # This does not change E[gradient] but reduces variance.
        r_mean = batch_returns_tensor.mean()
        r_std = batch_returns_tensor.std() + 1e-8
        r_normalized = (batch_returns_tensor - r_mean) / r_std

        # Sum log-probs across time steps for each trajectory, then weight
        # by the normalized return.  The negative sign converts gradient
        # ascent into a minimization problem for PyTorch.
        trajectory_logprob_sums = torch.stack(log_probs_per_step, dim=0).sum(dim=0)
        loss = -torch.mean(r_normalized * trajectory_logprob_sums)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        optimizer.step()

    return net, returns


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Set random seeds for numpy, torch CPU, and torch CUDA.

    Called before each ablation run in ``compare_reinforce_stability_2x2``
    (analysis.py) to ensure reproducible comparisons across configurations.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# REINFORCE with ablation toggles
# ---------------------------------------------------------------------------

def reinforce_with_options(
    config: Dict[str, object],
    num_episodes: int = 1000,
    K: int = 50,
    lr: float = 3e-4,
    arch: str = "medium",
    use_baseline: bool = True,
    use_clipping: bool = True,
    clip_value: float = 1.0,
) -> Tuple[nn.Module, List[float], List[float]]:
    """REINFORCE variant with independently toggleable baseline and clipping.

    This function is identical to ``reinforce`` except that baseline
    normalization and gradient clipping can each be turned on or off,
    enabling a 2x2 factorial ablation study.  It also records the
    pre-clipping gradient norm at each step for diagnostic plots.

    Parameters
    ----------
    use_baseline : bool
        If True, normalize returns by subtracting mean and dividing by std.
        If False, use raw returns as advantages.
    use_clipping : bool
        If True, clip the global gradient norm to ``clip_value``.
    clip_value : float
        Clipping threshold (only used when use_clipping=True).

    Returns
    -------
    net : MLPPolicyNet
    returns : list of float
    grad_norms : list of float
        Pre-clipping L2 norm of the full gradient vector at each update.

    Used in
    -------
    Called by ``compare_reinforce_stability_2x2`` (analysis.py), which is
    invoked in main.ipynb Section 6 -- REINFORCE Ablation.
    """
    device = _device(config)
    actions = list(config["ACTIONS"])

    net = build_policy_net(arch_name=arch, num_actions=len(actions), device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    returns: List[float] = []
    grad_norms: List[float] = []

    for episode in tqdm(range(num_episodes), desc=f"Ablation[{arch}] b={use_baseline} c={use_clipping}"):
        optimizer.zero_grad()
        log_probs_per_step, batch_returns_tensor = generate_trajectories_batch(net, K, config)

        avg_r = float(batch_returns_tensor.mean().item())
        returns.append(avg_r)
        trajectory_logprob_sums = torch.stack(log_probs_per_step, dim=0).sum(dim=0)

        if use_baseline:
            r_mean = batch_returns_tensor.mean()
            r_std = batch_returns_tensor.std() + 1e-8
            advantages = (batch_returns_tensor - r_mean) / r_std
        else:
            # Without baseline: raw returns used directly -- higher variance
            advantages = batch_returns_tensor

        loss = -torch.mean(advantages * trajectory_logprob_sums)
        loss.backward()

        # Record gradient norm *before* clipping for diagnostic comparison
        total_norm_sq = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total_norm_sq += float(p.grad.detach().pow(2).sum().item())
        grad_norms.append(total_norm_sq**0.5)

        if use_clipping:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
        optimizer.step()

    return net, returns, grad_norms
