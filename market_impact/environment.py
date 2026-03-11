"""Environment dynamics and action selection for the market-impact MDP.

This module defines the two lowest-level building blocks that every algorithm
in the package depends on:

* ``transition`` -- advances the MDP by one time step given a state and action.
* ``epsilon_greedy`` -- selects an action from a Q-table using the
  epsilon-greedy exploration strategy.

Both functions are stateless (no module-level globals) and receive all MDP
parameters through a ``config`` dictionary defined in the notebook.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def transition(
    state: Sequence[int],
    action: int,
    config: Dict[str, object],
) -> Tuple[List[int], float]:
    """Execute one step of the optimal-execution MDP.

    Given the current state (inventory, price, time) and the number of blocks
    to sell, compute the next state and the immediate reward.

    Parameters
    ----------
    state : (x, s, t)
        x = remaining inventory in blocks, s = discretized price state,
        t = current time index.
    action : int
        Number of blocks to sell (clipped to current inventory internally).
    config : dict
        MDP parameters.  Required keys: BLOCK_SIZE, NUM_S, dt, sigma, nu,
        mu, lmbda.

    Returns
    -------
    next_state : [x_next, s_next, t_next]
    reward : float
        R_t = n * a_t * S_t  -  lambda * n * X_{t+1}^2 * Var[S_{t+1}]

    Used in
    -------
    Called by ``q_learning``, ``sarsa`` (algorithms.py) and
    ``generate_trajectories_batch`` mirrors this logic in vectorized PyTorch.
    Indirectly used in every notebook section that trains or evaluates a policy.
    """
    block_size = int(config["BLOCK_SIZE"])
    num_s = int(config["NUM_S"])
    dt = float(config["dt"])
    sigma = float(config["sigma"])
    nu = float(config["nu"])
    mu = float(config["mu"])
    lmbda = float(config["lmbda"])

    x = int(state[0])
    s = int(state[1])
    t = int(state[2])

    # Cannot sell more blocks than currently held
    if action > x:
        action = x

    x_next = x - action
    t_next = t + int(dt)

    # Price dynamics: deterministic market-impact term + stochastic diffusion
    # S_{t+1} = S_t * exp(1 - nu * a_t) + sigma * S_t * sqrt(dt) * eps
    s_next = s * np.exp(1 - nu * action) + sigma * s * np.sqrt(dt) * np.random.randn()
    # Discretize to integer grid [0, NUM_S - 1]
    s_next = int(np.clip(np.ceil(s_next), 0, num_s - 1))

    # Var[S_{t+1}] from the log-normal model:
    #   S^2 * exp(2*mu*dt + sigma^2*dt) * (exp(sigma^2*dt) - 1)
    var = s_next**2 * np.exp(2 * mu * dt + sigma**2 * dt) * (np.exp(sigma**2 * dt) - 1)
    # Reward = execution revenue  -  risk penalty on remaining inventory
    reward = block_size * action * s - lmbda * block_size * (x_next**2) * var

    return [x_next, s_next, t_next], float(reward)


def epsilon_greedy(
    state: Sequence[int],
    q_value: np.ndarray,
    config: Dict[str, object],
    eps: float,
) -> int:
    """Select an action using the epsilon-greedy policy derived from a Q-table.

    With probability ``eps`` a random action is chosen uniformly; otherwise the
    greedy action argmax_a Q(x, s, t, a) is selected.  The result is always
    clipped so the agent cannot sell more blocks than it holds.

    Parameters
    ----------
    state : (x, s, t)
    q_value : ndarray of shape (NUM_BLOCKS, NUM_S, NUM_TIME_STEPS, |A|)
    config : dict
        Must contain the ``ACTIONS`` list.
    eps : float in [0, 1]
        Exploration probability.

    Returns
    -------
    action : int
        A valid action in {0, ..., min(max_action, x)}.

    Used in
    -------
    Called inside ``sarsa`` and ``q_learning`` (algorithms.py) at every
    in-episode step, and in ``estimate_state_visitation`` (analysis.py).
    """
    actions = list(config["ACTIONS"])

    if np.random.binomial(1, eps) == 1:
        action = int(np.random.choice(actions))
    else:
        values_ = q_value[int(state[0]), int(state[1]), int(state[2]), :]
        action = int(np.argmax(values_))

    # Clip: never sell more than current inventory
    if action > int(state[0]):
        action = int(state[0])

    return action
