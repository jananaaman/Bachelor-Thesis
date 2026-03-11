"""Comparison protocols, diagnostics, and post-training evaluation utilities.

This is the largest module in the package.  It contains:

**Statistical helpers** -- ``moving_average``, ``summarize_runs``,
``make_summary_table``, ``global_metrics``, ``plot_with_bands``.

**Comparison protocols** -- functions that train multiple algorithms over
several independent runs, aggregate learning curves, and produce publication-
ready plots and summary tables:

* ``compare_sarsa_qlearning`` -- basic SARSA vs Q-learning
* ``compare_all`` -- three-way (SARSA, Q-learning, REINFORCE)
* ``compare_fixed_updates_with_tables`` -- Protocol A (fixed parameter updates)
* ``compare_fixed_trajectories_with_tables`` -- Protocol B (fixed trajectory budget)
* ``compare_all_architectures`` -- architecture sweep (SARSA + Q-learning +
  REINFORCE-tiny/shallow/medium/deep)
* ``compare_reinforce_stability_2x2`` -- 2x2 ablation (baseline x clipping)

**Post-training evaluation** -- ``evaluate_policy``,
``run_post_training_evaluation``, ``plot_financial_diagnostics``,
``make_policy_heatmap``, ``twap_policy``, ``choose_best_reinforce_arch``,
``greedy_action_from_q``, ``greedy_action_from_net``.

**Deep-dive diagnostics** (main.ipynb Section 10) --
``tabular_lr_sensitivity``, ``plot_q_table_diagnostics``,
``estimate_state_visitation``, ``plot_reinforce_loss_landscape``.

All functions receive MDP parameters through a ``config`` dict.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colors as mcolors
from tqdm.auto import tqdm

from .algorithms import (
    generate_trajectories_batch,
    make_input,
    q_learning,
    reinforce,
    reinforce_with_options,
    sarsa,
    set_all_seeds,
)
from .environment import epsilon_greedy, transition


# ===================================================================
# Statistical helpers
# ===================================================================

def moving_average(x: Sequence[float], window: int) -> np.ndarray:
    """Compute a simple moving average with a uniform kernel.

    Parameters
    ----------
    x : array-like
        1-D signal.
    window : int
        Kernel width.  If <= 1 the input is returned unchanged.

    Returns
    -------
    ndarray of length ``len(x) - window + 1`` (mode="valid").

    Used in
    -------
    Smoothed-curve plots inside every ``compare_*`` function and in
    ``tabular_lr_sensitivity``.
    """
    arr = np.asarray(x, dtype=float)
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def summarize_runs(run_curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate multiple independent learning curves into mean +/- spread.

    Parameters
    ----------
    run_curves : ndarray of shape (n_runs, T)
        Each row is one run's per-episode return.

    Returns
    -------
    mean : ndarray (T,)
    std : ndarray (T,)
        Sample standard deviation (ddof=1).
    stderr : ndarray (T,)
        Standard error of the mean = std / sqrt(n_runs).

    Used in
    -------
    Called inside every ``compare_*`` function to turn raw per-run curves
    into the statistics that ``plot_with_bands`` renders.
    """
    run_curves = np.asarray(run_curves, dtype=float)
    mean = run_curves.mean(axis=0)
    if run_curves.shape[0] > 1:
        std = run_curves.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    stderr = std / np.sqrt(run_curves.shape[0]) if run_curves.shape[0] > 0 else std
    return mean, std, stderr


def make_summary_table(
    name: str,
    mean: np.ndarray,
    std: np.ndarray,
    stderr: np.ndarray,
    x_values: np.ndarray,
    checkpoints: Iterable[int],
) -> pd.DataFrame:
    """Extract statistics at specific checkpoint indices into a DataFrame.

    Useful for quoting exact numbers in thesis text (e.g. "at episode 500,
    SARSA achieves a mean return of ...").

    Parameters
    ----------
    name : str
        Algorithm label for the "Algorithm" column.
    mean, std, stderr : ndarray
        Output of ``summarize_runs``.
    x_values : ndarray
        The x-axis values (episodes or trajectories) corresponding to each
        index in ``mean``.
    checkpoints : iterable of int
        Indices to extract.  Use -1 for the final index.

    Returns
    -------
    DataFrame with columns: Algorithm, Index, x_value, Mean, Std, StdErr.
    """
    rows = []
    t_len = len(mean)
    for c in checkpoints:
        idx = (t_len - 1) if c == -1 else c
        idx = int(np.clip(idx, 0, t_len - 1))
        rows.append(
            {
                "Algorithm": name,
                "Index": idx,
                "x_value": float(x_values[idx]),
                "Mean": float(mean[idx]),
                "Std": float(std[idx]),
                "StdErr": float(stderr[idx]),
            }
        )
    return pd.DataFrame(rows)


def global_metrics(
    name: str,
    mean: np.ndarray,
    std: np.ndarray,
    stderr: np.ndarray,
) -> Dict[str, float | int | str]:
    """Compute a single-row summary of an entire learning curve.

    Metrics: FinalMean, FinalStd, FinalStdErr, BestIdx, BestMean, AUC(mean).

    AUC is the area under the mean curve (trapezoidal rule) and provides a
    single number summarizing overall learning quality -- an algorithm that
    learns faster *and* converges higher will have a larger AUC.

    Used in
    -------
    Called inside every ``compare_*`` function to build the "global" summary
    table displayed in the notebook.
    """
    final_idx = len(mean) - 1
    best_idx = int(np.argmax(mean))

    # numpy >= 2.0 renamed trapz -> trapezoid; handle both
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(mean))
    elif hasattr(np, "trapz"):
        auc = float(np.trapz(mean))
    else:
        auc = float(np.sum((mean[:-1] + mean[1:]) * 0.5)) if len(mean) > 1 else float(mean[0])

    return {
        "Algorithm": name,
        "FinalMean": float(mean[final_idx]),
        "FinalStd": float(std[final_idx]),
        "FinalStdErr": float(stderr[final_idx]),
        "BestIdx": best_idx,
        "BestMean": float(mean[best_idx]),
        "AUC(mean)": auc,
    }


def plot_with_bands(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    stderr: np.ndarray,
    label: str,
    band: str = "std",
) -> None:
    """Plot a mean curve with a shaded uncertainty band.

    Parameters
    ----------
    band : {"std", "stderr", "none"}
        Which spread measure to use for the shaded region.

    Used in
    -------
    Called by every ``compare_*`` function for the raw (non-smoothed) plots.
    """
    plt.plot(x, mean, label=label)
    if band == "std":
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    elif band == "stderr":
        plt.fill_between(x, mean - stderr, mean + stderr, alpha=0.2)
    elif band == "none":
        pass
    else:
        raise ValueError("band must be one of: 'std', 'stderr', 'none'")


# ===================================================================
# Comparison protocols
# ===================================================================

def compare_sarsa_qlearning(
    config: Dict[str, object],
    num_episodes: int = 1000,
    num_runs: int = 100,
    epoch_size: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train SARSA and Q-learning side-by-side and plot average returns.

    This is the simplest comparison: no REINFORCE, no statistical bands --
    just the mean return curve averaged over ``num_runs`` independent runs.

    Parameters
    ----------
    epoch_size : int
        Exploration rate is decayed every ``epoch_size`` episodes:
        eps_u = eps_0 * (1 - eps_0)^(u // epoch_size).

    Returns
    -------
    q_table_sarsa, q_table_qlearning : ndarray
        The Q-tables from the *last* run (useful for quick inspection but
        not statistically representative).

    Used in
    -------
    Not currently called in main.ipynb (superseded by
    ``compare_all_architectures``), but available for quick sanity checks.
    """
    avg_rewards_sarsa = np.zeros(num_episodes)
    avg_rewards_qlearning = np.zeros(num_episodes)

    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])
    num_time_steps = int(config["NUM_TIME_STEPS"])
    actions = list(config["ACTIONS"])
    epsilon = float(config["epsilon"])

    for _ in tqdm(range(num_runs)):
        q_table_sarsa = np.zeros((num_blocks, num_s, num_time_steps, len(actions)))
        q_table_qlearning = np.copy(q_table_sarsa)

        for episode in range(num_episodes):
            # Exponential decay of exploration rate within each epoch
            eps = epsilon * ((1 - epsilon) ** (episode // epoch_size))
            avg_rewards_sarsa[episode] += sarsa(q_table_sarsa, config, eps=eps)
            avg_rewards_qlearning[episode] += q_learning(q_table_qlearning, config, eps=eps)

    avg_rewards_sarsa /= num_runs
    avg_rewards_qlearning /= num_runs

    plt.plot(avg_rewards_sarsa, label="SARSA")
    plt.plot(avg_rewards_qlearning, label="Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Average return per episode")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return q_table_sarsa, q_table_qlearning


def compare_all(
    config: Dict[str, object],
    num_episodes: int = 1000,
    num_runs: int = 100,
    num_pg_runs: int = 10,
    epoch_size: int = 25,
    K: int = 50,
    lr: float = 3e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-way comparison: SARSA vs Q-learning vs REINFORCE (default arch).

    Returns the mean return curves (averaged over runs) for each algorithm.
    Note: this is *not* trajectory-fair -- REINFORCE uses K trajectories per
    update while tabular methods use 1.

    Used in
    -------
    Not currently called in main.ipynb (superseded by
    ``compare_all_architectures``).
    """
    avg_rewards_sarsa = np.zeros(num_episodes)
    avg_rewards_qlearning = np.zeros(num_episodes)

    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])
    num_time_steps = int(config["NUM_TIME_STEPS"])
    actions = list(config["ACTIONS"])
    epsilon = float(config["epsilon"])

    for _ in tqdm(range(num_runs), desc="SARSA & Q-Learning runs"):
        q_table_sarsa = np.zeros((num_blocks, num_s, num_time_steps, len(actions)))
        q_table_qlearning = np.copy(q_table_sarsa)

        for episode in range(num_episodes):
            eps = epsilon * ((1 - epsilon) ** (episode // epoch_size))
            avg_rewards_sarsa[episode] += sarsa(q_table_sarsa, config, eps=eps)
            avg_rewards_qlearning[episode] += q_learning(q_table_qlearning, config, eps=eps)

    avg_rewards_sarsa /= num_runs
    avg_rewards_qlearning /= num_runs

    avg_rewards_reinforce = np.zeros(num_episodes)
    for _ in tqdm(range(num_pg_runs), desc="REINFORCE runs"):
        _, reinforce_returns = reinforce(config, num_episodes=num_episodes, K=K, lr=lr)
        avg_rewards_reinforce += np.asarray(reinforce_returns)
    avg_rewards_reinforce /= num_pg_runs

    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards_sarsa, label="SARSA", alpha=0.8)
    plt.plot(avg_rewards_qlearning, label="Q-Learning", alpha=0.8)
    plt.plot(avg_rewards_reinforce, label="REINFORCE", alpha=0.8)
    plt.xlabel("Episodes")
    plt.ylabel("Average return per episode")
    plt.title("SARSA vs Q-Learning vs REINFORCE - Market Impact MDP")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return avg_rewards_sarsa, avg_rewards_qlearning, avg_rewards_reinforce


def compare_fixed_trajectories_with_tables(
    config: Dict[str, object],
    total_trajectories: int = 20000,
    num_runs: int = 10,
    num_pg_runs: int = 5,
    epoch_size: int = 25,
    K: int = 50,
    lr: float = 3e-4,
    band: str = "std",
    plot_smooth: int = 1,
    smooth_window_tab: int = 200,
    smooth_window_pg: int | None = None,
    smooth_separate_plot: int = 1,
    checkpoints: Sequence[int] = (0, 1000, 5000, -1),
    export_csv_prefix: str | None = None,
) -> Dict[str, object]:
    """Protocol B: compare algorithms under a fixed total trajectory budget.

    All three algorithms consume the same number of environment trajectories:
    tabular methods run ``total_trajectories`` episodes (1 trajectory each),
    REINFORCE runs ``total_trajectories // K`` episodes of K trajectories.

    This gives a **sample-efficiency** comparison: the x-axis is the
    cumulative number of trajectories used.

    Produces two plots (raw + smoothed) and three summary tables (checkpoint
    tabular, checkpoint PG, global metrics).

    Returns
    -------
    dict with keys:
        "runs"    -- raw per-run curves {sarsa, q, pg}
        "mean"    -- aggregated mean curves
        "std"     -- aggregated std curves
        "stderr"  -- aggregated stderr curves
        "tables"  -- {"checkpoints_tabular", "checkpoints_pg", "global"}

    Used in
    -------
    main.ipynb Section 8 -- Fixed Trajectory Budget Comparison.
    """
    del smooth_separate_plot  # kept for API compat, unused

    episodes_tab = total_trajectories
    sarsa_runs = np.zeros((num_runs, episodes_tab))
    q_runs = np.zeros((num_runs, episodes_tab))

    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])
    num_time_steps = int(config["NUM_TIME_STEPS"])
    actions = list(config["ACTIONS"])
    epsilon = float(config["epsilon"])

    for r in tqdm(range(num_runs), desc="TABULAR runs (fixed trajectories)"):
        q_sarsa = np.zeros((num_blocks, num_s, num_time_steps, len(actions)))
        q_q = np.zeros_like(q_sarsa)
        for ep in range(episodes_tab):
            eps = epsilon * ((1 - epsilon) ** (ep // epoch_size))
            sarsa_runs[r, ep] = sarsa(q_sarsa, config, eps=eps)
            q_runs[r, ep] = q_learning(q_q, config, eps=eps)

    sarsa_mean, sarsa_std, sarsa_stderr = summarize_runs(sarsa_runs)
    q_mean, q_std, q_stderr = summarize_runs(q_runs)

    # REINFORCE uses K trajectories per update, so fewer episodes
    episodes_pg = total_trajectories // K
    pg_runs = np.zeros((num_pg_runs, episodes_pg))
    for r in tqdm(range(num_pg_runs), desc="REINFORCE runs (fixed trajectories)"):
        _, returns = reinforce(config, num_episodes=episodes_pg, K=K, lr=lr)
        pg_runs[r, :] = np.asarray(returns)

    pg_mean, pg_std, pg_stderr = summarize_runs(pg_runs)
    x_tab = np.arange(episodes_tab)
    # Map REINFORCE episodes to the equivalent trajectory count for a fair x-axis
    x_pg = np.linspace(0, total_trajectories, episodes_pg)

    plt.figure(figsize=(10, 5))
    plot_with_bands(x_tab, sarsa_mean, sarsa_std, sarsa_stderr, "SARSA", band=band)
    plot_with_bands(x_tab, q_mean, q_std, q_stderr, "Q-learning", band=band)
    plot_with_bands(x_pg, pg_mean, pg_std, pg_stderr, "REINFORCE", band=band)
    plt.xlabel("Total trajectories used")
    plt.ylabel("Return")
    plt.title(f"Fixed Trajectory Budget (mean with {band} band)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if plot_smooth:
        if smooth_window_pg is None:
            smooth_window_pg = max(1, smooth_window_tab // K)

        sarsa_s = moving_average(sarsa_mean, smooth_window_tab)
        q_s = moving_average(q_mean, smooth_window_tab)
        pg_s = moving_average(pg_mean, smooth_window_pg)
        x_tab_s = x_tab[smooth_window_tab - 1 :]
        x_pg_s = x_pg[smooth_window_pg - 1 :]

        plt.figure(figsize=(10, 5))
        plt.plot(x_tab_s, sarsa_s, label=f"SARSA (MA {smooth_window_tab})")
        plt.plot(x_tab_s, q_s, label=f"Q-learning (MA {smooth_window_tab})")
        plt.plot(x_pg_s, pg_s, label=f"REINFORCE (MA {smooth_window_pg})")
        plt.xlabel("Total trajectories used")
        plt.ylabel("Return")
        plt.title("Fixed Trajectory Budget (smoothed mean curves)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    tab_table = pd.concat(
        [
            make_summary_table("SARSA", sarsa_mean, sarsa_std, sarsa_stderr, x_tab, checkpoints),
            make_summary_table("Q-learning", q_mean, q_std, q_stderr, x_tab, checkpoints),
        ],
        ignore_index=True,
    )
    pg_table = make_summary_table("REINFORCE", pg_mean, pg_std, pg_stderr, x_pg, checkpoints)

    global_table = pd.DataFrame(
        [
            global_metrics("SARSA", sarsa_mean, sarsa_std, sarsa_stderr),
            global_metrics("Q-learning", q_mean, q_std, q_stderr),
            global_metrics("REINFORCE", pg_mean, pg_std, pg_stderr),
        ]
    )

    if export_csv_prefix:
        tab_table.to_csv(f"{export_csv_prefix}_checkpoints_tabular.csv", index=False)
        pg_table.to_csv(f"{export_csv_prefix}_checkpoints_pg.csv", index=False)
        global_table.to_csv(f"{export_csv_prefix}_global_metrics.csv", index=False)

    return {
        "runs": {"sarsa": sarsa_runs, "q": q_runs, "pg": pg_runs},
        "mean": {"sarsa": sarsa_mean, "q": q_mean, "pg": pg_mean},
        "std": {"sarsa": sarsa_std, "q": q_std, "pg": pg_std},
        "stderr": {"sarsa": sarsa_stderr, "q": q_stderr, "pg": pg_stderr},
        "tables": {"checkpoints_tabular": tab_table, "checkpoints_pg": pg_table, "global": global_table},
    }


def compare_fixed_updates_with_tables(
    config: Dict[str, object],
    num_updates: int = 2000,
    num_runs: int = 10,
    num_pg_runs: int = 5,
    epoch_size: int = 25,
    K: int = 50,
    lr: float = 3e-4,
    band: str = "std",
    plot_smooth: int = 1,
    smooth_window: int = 100,
    smooth_separate_plot: int = 1,
    checkpoints: Sequence[int] = (0, 100, 500, -1),
    export_csv_prefix: str | None = None,
) -> Dict[str, object]:
    """Protocol A: compare algorithms under a fixed number of parameter updates.

    All three algorithms perform the same number of gradient/table updates.
    Note that REINFORCE uses K trajectories per update, so this protocol is
    **not** trajectory-fair -- it answers "who learns fastest per update?"

    Returns
    -------
    Same structure as ``compare_fixed_trajectories_with_tables``.

    Used in
    -------
    Available for use in the notebook but not in the current default
    storyline (Protocol B is preferred for the thesis).
    """
    del smooth_separate_plot

    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])
    num_time_steps = int(config["NUM_TIME_STEPS"])
    actions = list(config["ACTIONS"])
    epsilon = float(config["epsilon"])

    sarsa_runs = np.zeros((num_runs, num_updates))
    q_runs = np.zeros((num_runs, num_updates))

    for r in tqdm(range(num_runs), desc="TABULAR runs (fixed updates)"):
        q_sarsa = np.zeros((num_blocks, num_s, num_time_steps, len(actions)))
        q_q = np.zeros_like(q_sarsa)
        for u in range(num_updates):
            eps = epsilon * ((1 - epsilon) ** (u // epoch_size))
            sarsa_runs[r, u] = sarsa(q_sarsa, config, eps=eps)
            q_runs[r, u] = q_learning(q_q, config, eps=eps)

    sarsa_mean, sarsa_std, sarsa_stderr = summarize_runs(sarsa_runs)
    q_mean, q_std, q_stderr = summarize_runs(q_runs)

    pg_runs = np.zeros((num_pg_runs, num_updates))
    for r in tqdm(range(num_pg_runs), desc="REINFORCE runs (fixed updates)"):
        _, returns = reinforce(config, num_episodes=num_updates, K=K, lr=lr)
        pg_runs[r, :] = np.asarray(returns)
    pg_mean, pg_std, pg_stderr = summarize_runs(pg_runs)

    x = np.arange(num_updates)
    plt.figure(figsize=(10, 5))
    plot_with_bands(x, sarsa_mean, sarsa_std, sarsa_stderr, "SARSA", band=band)
    plot_with_bands(x, q_mean, q_std, q_stderr, "Q-learning", band=band)
    plot_with_bands(x, pg_mean, pg_std, pg_stderr, "REINFORCE", band=band)
    plt.xlabel("Parameter updates")
    plt.ylabel("Return")
    plt.title(f"Fixed Updates (mean with {band} band)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if plot_smooth:
        sarsa_s = moving_average(sarsa_mean, smooth_window)
        q_s = moving_average(q_mean, smooth_window)
        pg_s = moving_average(pg_mean, smooth_window)
        x_s = x[smooth_window - 1 :]

        plt.figure(figsize=(10, 5))
        plt.plot(x_s, sarsa_s, label=f"SARSA (MA {smooth_window})")
        plt.plot(x_s, q_s, label=f"Q-learning (MA {smooth_window})")
        plt.plot(x_s, pg_s, label=f"REINFORCE (MA {smooth_window})")
        plt.xlabel("Parameter updates")
        plt.ylabel("Return")
        plt.title("Fixed Updates (smoothed mean curves)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    checkpoints_table = pd.concat(
        [
            make_summary_table("SARSA", sarsa_mean, sarsa_std, sarsa_stderr, x, checkpoints),
            make_summary_table("Q-learning", q_mean, q_std, q_stderr, x, checkpoints),
            make_summary_table("REINFORCE", pg_mean, pg_std, pg_stderr, x, checkpoints),
        ],
        ignore_index=True,
    )
    global_table = pd.DataFrame(
        [
            global_metrics("SARSA", sarsa_mean, sarsa_std, sarsa_stderr),
            global_metrics("Q-learning", q_mean, q_std, q_stderr),
            global_metrics("REINFORCE", pg_mean, pg_std, pg_stderr),
        ]
    )

    if export_csv_prefix:
        checkpoints_table.to_csv(f"{export_csv_prefix}_checkpoints.csv", index=False)
        global_table.to_csv(f"{export_csv_prefix}_global_metrics.csv", index=False)

    return {
        "runs": {"sarsa": sarsa_runs, "q": q_runs, "pg": pg_runs},
        "mean": {"sarsa": sarsa_mean, "q": q_mean, "pg": pg_mean},
        "std": {"sarsa": sarsa_std, "q": q_std, "pg": pg_std},
        "stderr": {"sarsa": sarsa_stderr, "q": q_stderr, "pg": pg_stderr},
        "tables": {"checkpoints": checkpoints_table, "global": global_table},
    }


def compare_all_architectures(
    config: Dict[str, object],
    num_episodes: int = 1000,
    num_runs: int = 20,
    num_pg_runs: int = 8,
    epoch_size: int = 25,
    K: int = 50,
    lr: float = 3e-4,
    reinforce_archs: Sequence[str] = ("tiny", "shallow", "medium", "deep"),
    band: str = "std",
    smooth_window: int = 50,
    plot_smooth: bool = True,
    checkpoints: Sequence[int] = (0, 100, 500, -1),
    export_csv_prefix: str | None = None,
) -> Dict[str, object]:
    """Full architecture sweep: SARSA + Q-learning + multiple REINFORCE archs.

    Trains all algorithms for the same number of episodes and produces:
    1. A raw plot with uncertainty bands
    2. A smoothed-mean overlay plot
    3. Checkpoint and global summary tables

    Importantly, this function also **retains the trained models** (the last
    Q-tables and the last REINFORCE net for each architecture) so they can be
    passed to ``run_post_training_evaluation`` for policy-level analysis.

    Returns
    -------
    dict with keys:
        "runs"    -- raw per-run curves
        "mean", "std", "stderr" -- aggregated statistics
        "trained" -- {"q_sarsa", "q_qlearning", "reinforce_nets"}
        "tables"  -- {"checkpoints", "global"}

    Used in
    -------
    main.ipynb Section 7 -- Architecture Comparison.  Its output is then
    passed to ``run_post_training_evaluation`` in Section 9 and to the
    deep-dive diagnostics in Section 10.
    """
    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])
    num_time_steps = int(config["NUM_TIME_STEPS"])
    actions = list(config["ACTIONS"])
    epsilon = float(config["epsilon"])

    sarsa_runs = np.zeros((num_runs, num_episodes))
    q_runs = np.zeros((num_runs, num_episodes))
    final_q_sarsa = None
    final_q_qlearning = None

    for r in tqdm(range(num_runs), desc="TABULAR runs"):
        q_sarsa = np.zeros((num_blocks, num_s, num_time_steps, len(actions)))
        q_q = np.zeros_like(q_sarsa)
        for ep in range(num_episodes):
            eps = epsilon * ((1 - epsilon) ** (ep // epoch_size))
            sarsa_runs[r, ep] = sarsa(q_sarsa, config, eps=eps)
            q_runs[r, ep] = q_learning(q_q, config, eps=eps)
        # Keep the last run's Q-tables for post-training evaluation
        final_q_sarsa = q_sarsa
        final_q_qlearning = q_q

    sarsa_mean, sarsa_std, sarsa_stderr = summarize_runs(sarsa_runs)
    q_mean, q_std, q_stderr = summarize_runs(q_runs)

    pg_runs_by_arch: Dict[str, np.ndarray] = {}
    pg_stats_by_arch: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    final_pg_nets: Dict[str, torch.nn.Module] = {}

    for arch in reinforce_archs:
        arch_runs = np.zeros((num_pg_runs, num_episodes))
        last_net = None
        for r in tqdm(range(num_pg_runs), desc=f"REINFORCE-{arch} runs"):
            net, returns = reinforce(config, num_episodes=num_episodes, K=K, lr=lr, arch=arch)
            arch_runs[r, :] = np.asarray(returns)
            last_net = net
        pg_runs_by_arch[arch] = arch_runs
        pg_stats_by_arch[arch] = summarize_runs(arch_runs)
        if last_net is not None:
            final_pg_nets[arch] = last_net

    # --- Plot 1: raw curves with bands ---
    x = np.arange(num_episodes)
    plt.figure(figsize=(12, 6))
    plot_with_bands(x, sarsa_mean, sarsa_std, sarsa_stderr, "SARSA", band=band)
    plot_with_bands(x, q_mean, q_std, q_stderr, "Q-learning", band=band)
    for arch in reinforce_archs:
        mean_, std_, stderr_ = pg_stats_by_arch[arch]
        plot_with_bands(x, mean_, std_, stderr_, f"REINFORCE-{arch}", band=band)
    plt.xlabel("Episodes / Updates")
    plt.ylabel("Return")
    plt.title(f"All Algorithms Comparison (band={band})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 2: smoothed means ---
    if plot_smooth and smooth_window > 1:
        plt.figure(figsize=(12, 6))
        x_s = x[smooth_window - 1 :]
        plt.plot(x_s, moving_average(sarsa_mean, smooth_window), label=f"SARSA (MA {smooth_window})")
        plt.plot(x_s, moving_average(q_mean, smooth_window), label=f"Q-learning (MA {smooth_window})")
        for arch in reinforce_archs:
            mean_, _, _ = pg_stats_by_arch[arch]
            plt.plot(x_s, moving_average(mean_, smooth_window), label=f"REINFORCE-{arch} (MA {smooth_window})")
        plt.xlabel("Episodes / Updates")
        plt.ylabel("Return")
        plt.title("All Algorithms Comparison (smoothed means)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Summary tables ---
    checkpoint_tables = [
        make_summary_table("SARSA", sarsa_mean, sarsa_std, sarsa_stderr, x, checkpoints),
        make_summary_table("Q-learning", q_mean, q_std, q_stderr, x, checkpoints),
    ]
    global_rows: List[Dict[str, float | int | str]] = [
        global_metrics("SARSA", sarsa_mean, sarsa_std, sarsa_stderr),
        global_metrics("Q-learning", q_mean, q_std, q_stderr),
    ]
    for arch in reinforce_archs:
        mean_, std_, stderr_ = pg_stats_by_arch[arch]
        algo_name = f"REINFORCE-{arch}"
        checkpoint_tables.append(make_summary_table(algo_name, mean_, std_, stderr_, x, checkpoints))
        global_rows.append(global_metrics(algo_name, mean_, std_, stderr_))

    checkpoints_table = pd.concat(checkpoint_tables, ignore_index=True)
    global_table = pd.DataFrame(global_rows)

    if export_csv_prefix:
        checkpoints_table.to_csv(f"{export_csv_prefix}_checkpoints.csv", index=False)
        global_table.to_csv(f"{export_csv_prefix}_global_metrics.csv", index=False)

    return {
        "runs": {"sarsa": sarsa_runs, "qlearning": q_runs, "reinforce": pg_runs_by_arch},
        "mean": {"sarsa": sarsa_mean, "qlearning": q_mean, "reinforce": {k: v[0] for k, v in pg_stats_by_arch.items()}},
        "std": {"sarsa": sarsa_std, "qlearning": q_std, "reinforce": {k: v[1] for k, v in pg_stats_by_arch.items()}},
        "stderr": {
            "sarsa": sarsa_stderr,
            "qlearning": q_stderr,
            "reinforce": {k: v[2] for k, v in pg_stats_by_arch.items()},
        },
        "trained": {"q_sarsa": final_q_sarsa, "q_qlearning": final_q_qlearning, "reinforce_nets": final_pg_nets},
        "tables": {"checkpoints": checkpoints_table, "global": global_table},
    }


# ===================================================================
# Post-training policy evaluation
# ===================================================================

def greedy_action_from_q(q_table: np.ndarray, state: Sequence[int]) -> int:
    """Return the greedy action argmax_a Q(x, s, t, a), clipped by inventory.

    Used in
    -------
    ``run_post_training_evaluation`` wraps this into a lambda to create a
    deterministic policy from a trained Q-table.
    """
    x, s, t = int(state[0]), int(state[1]), int(state[2])
    q_vals = q_table[x, s, t, :]
    action = int(np.argmax(q_vals))
    return min(action, x)


def greedy_action_from_net(net: torch.nn.Module, state: Sequence[int], config: Dict[str, object]) -> int:
    """Return the greedy action from a trained REINFORCE policy network.

    Builds a normalized input via ``make_input``, runs a forward pass, masks
    invalid actions, and returns argmax.  No gradient computation.

    Used in
    -------
    ``run_post_training_evaluation`` wraps this into a lambda to create a
    deterministic policy from the best trained network.
    """
    x, s, t = int(state[0]), int(state[1]), int(state[2])
    inp = make_input(t, x, s, config).unsqueeze(0)  # shape (1, 3)
    actions = list(config["ACTIONS"])
    actions_tensor = torch.tensor(actions, device=config["DEVICE"])  # type: ignore[arg-type]

    with torch.no_grad():
        logits = net(inp).squeeze(0)
        # Mask out actions that would sell more than current inventory
        valid_mask = actions_tensor <= int(x)
        masked_logits = logits.masked_fill(~valid_mask, -1e9)
        action_idx = int(torch.argmax(masked_logits).item())
    action = int(actions[action_idx])
    return min(action, x)


def evaluate_policy(
    policy_fn: Callable[[List[int]], int],
    config: Dict[str, object],
    num_episodes: int = 1000,
) -> Dict[str, object]:
    """Roll out a deterministic policy for many episodes and collect statistics.

    Parameters
    ----------
    policy_fn : callable
        Maps a state [x, s, t] to an action (int).

    Returns
    -------
    dict with keys:
        "returns"         -- ndarray (num_episodes,)
        "inventory_paths" -- ndarray (num_episodes, T)
        "price_paths"     -- ndarray (num_episodes, T)
        "action_paths"    -- ndarray (num_episodes, T-1)
        "mean_inventory"  -- ndarray (T,)
        "mean_price"      -- ndarray (T,)
        "mean_action"     -- ndarray (T-1,)
        "std_return"      -- float
        "mean_return"     -- float

    Used in
    -------
    Called by ``run_post_training_evaluation`` for each policy (SARSA,
    Q-learning, REINFORCE, TWAP).
    """
    num_time_steps = int(config["NUM_TIME_STEPS"])
    start = [int(config["NUM_BLOCKS"]) - 1, int(config["S0"]), 0]

    episode_returns = np.zeros(num_episodes)
    inventory_paths = np.zeros((num_episodes, num_time_steps))
    price_paths = np.zeros((num_episodes, num_time_steps))
    action_paths = np.zeros((num_episodes, num_time_steps - 1))

    for ep in range(num_episodes):
        state = start.copy()
        total_r = 0.0

        inventory_paths[ep, 0] = state[0]
        price_paths[ep, 0] = state[1]

        for t in range(num_time_steps - 1):
            action = int(policy_fn(state))
            action = min(action, state[0])
            next_state, reward = transition(state, action, config)
            total_r += reward

            action_paths[ep, t] = action
            inventory_paths[ep, t + 1] = next_state[0]
            price_paths[ep, t + 1] = next_state[1]
            state = next_state

        episode_returns[ep] = total_r

    return {
        "returns": episode_returns,
        "inventory_paths": inventory_paths,
        "price_paths": price_paths,
        "action_paths": action_paths,
        "mean_inventory": inventory_paths.mean(axis=0),
        "mean_price": price_paths.mean(axis=0),
        "mean_action": action_paths.mean(axis=0),
        "std_return": episode_returns.std(ddof=1) if num_episodes > 1 else 0.0,
        "mean_return": episode_returns.mean(),
    }


def choose_best_reinforce_arch(global_table: pd.DataFrame) -> str:
    """Pick the REINFORCE architecture with the highest FinalMean from a global table.

    Scans rows whose Algorithm column starts with "REINFORCE-" and returns
    the architecture suffix (e.g. "medium").

    Used in
    -------
    ``run_post_training_evaluation`` and main.ipynb Section 10 (loss
    landscape) to select the best-performing network for evaluation.
    """
    tbl = global_table.copy()
    reinforce_tbl = tbl[tbl["Algorithm"].str.startswith("REINFORCE-")]
    if len(reinforce_tbl) == 0:
        raise ValueError("No REINFORCE architecture rows found in global_table")
    best_idx = reinforce_tbl["FinalMean"].idxmax()
    best_name = reinforce_tbl.loc[best_idx, "Algorithm"]
    return best_name.replace("REINFORCE-", "")


def twap_policy(state: Sequence[int], config: Dict[str, object]) -> int:
    """Time-Weighted Average Price: sell inventory uniformly over remaining time.

    a_t = ceil(X_t / remaining_steps).  This is a model-free benchmark that
    ignores price state entirely.

    Used in
    -------
    ``run_post_training_evaluation`` includes TWAP as a baseline when
    ``include_twap=True`` (main.ipynb Section 9).
    """
    x, _, t = int(state[0]), int(state[1]), int(state[2])
    num_time_steps = int(config["NUM_TIME_STEPS"])
    remaining_steps = max(1, (num_time_steps - 1) - t)
    a = int(np.ceil(x / remaining_steps))
    return int(np.clip(a, 0, x))


def make_policy_heatmap(
    policy_fn: Callable[[List[int]], int],
    config: Dict[str, object],
    t_fixed: int = 0,
) -> np.ndarray:
    """Build a 2-D heatmap of actions over (inventory x, price state s) at a fixed time.

    Returns an ndarray of shape (NUM_BLOCKS, NUM_S) where entry [x, s] is the
    action chosen by ``policy_fn([x, s, t_fixed])``.

    Used in
    -------
    Called by ``plot_financial_diagnostics`` to produce the policy heatmap
    panel in the evaluation plots (main.ipynb Section 9).
    """
    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])
    hmap = np.zeros((num_blocks, num_s))
    for x in range(num_blocks):
        for s in range(num_s):
            hmap[x, s] = policy_fn([x, s, t_fixed])
    return hmap


def plot_financial_diagnostics(eval_dict: Dict[str, Dict[str, object]], t_fixed_for_heatmap: int = 0) -> None:
    """Produce a full suite of financial diagnostic plots for evaluated policies.

    Generates five figures:
    1. Return distribution histograms + boxplot
    2. Average inventory liquidation profile over time
    3. Average trading rate (blocks sold) per step
    4. Average discretized price trajectory under each policy
    5. Side-by-side policy heatmaps (action as a function of x and s)

    Parameters
    ----------
    eval_dict : dict
        Keys are policy names (e.g. "SARSA", "Q-learning"), values are the
        dicts returned by ``evaluate_policy`` augmented with "policy_fn" and
        "config" entries.

    Used in
    -------
    Called by ``run_post_training_evaluation`` (main.ipynb Section 9).
    """
    labels = list(eval_dict.keys())

    # --- Return distribution ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for name in labels:
        plt.hist(eval_dict[name]["returns"], bins=40, alpha=0.35, label=name)
    plt.xlabel("Episode return")
    plt.ylabel("Frequency")
    plt.title("Return distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot([eval_dict[name]["returns"] for name in labels], tick_labels=labels, showfliers=False)
    plt.ylabel("Episode return")
    plt.title("Return boxplot")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    # --- Inventory liquidation profile ---
    plt.figure(figsize=(8, 5))
    for name in labels:
        plt.plot(eval_dict[name]["mean_inventory"], label=name)
    plt.xlabel("Time step")
    plt.ylabel("Average remaining inventory")
    plt.title("Inventory liquidation profile")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Trading rate ---
    plt.figure(figsize=(8, 5))
    for name in labels:
        plt.plot(eval_dict[name]["mean_action"], label=name)
    plt.xlabel("Time step")
    plt.ylabel("Average action (blocks sold)")
    plt.title("Average trading rate per step")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Price trajectory ---
    plt.figure(figsize=(8, 5))
    for name in labels:
        plt.plot(eval_dict[name]["mean_price"], label=name)
    plt.xlabel("Time step")
    plt.ylabel("Average discretized price state")
    plt.title("Average price trajectory under policy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Policy heatmaps ---
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for j, name in enumerate(labels):
        hmap = make_policy_heatmap(eval_dict[name]["policy_fn"], eval_dict[name]["config"], t_fixed=t_fixed_for_heatmap)
        im = axes[0, j].imshow(hmap, origin="lower", aspect="auto", cmap="viridis")
        axes[0, j].set_title(f"{name} | t={t_fixed_for_heatmap}")
        axes[0, j].set_xlabel("Price state s")
        axes[0, j].set_ylabel("Inventory x")
        fig.colorbar(im, ax=axes[0, j], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def run_post_training_evaluation(
    comparison_out: Dict[str, object],
    config: Dict[str, object],
    num_eval_episodes: int = 2000,
    include_twap: bool = True,
) -> Dict[str, object]:
    """Evaluate all trained policies on fresh episodes and produce diagnostics.

    Takes the output of ``compare_all_architectures`` (which contains the
    trained Q-tables and REINFORCE networks), extracts greedy policies from
    each, and runs ``evaluate_policy`` + ``plot_financial_diagnostics``.

    The best REINFORCE architecture is selected automatically via
    ``choose_best_reinforce_arch``.

    Parameters
    ----------
    comparison_out : dict
        Output of ``compare_all_architectures``.
    include_twap : bool
        If True, also evaluate the TWAP benchmark.

    Returns
    -------
    dict with keys:
        "best_arch"   -- str, name of the best REINFORCE architecture
        "evaluations" -- dict mapping policy name -> evaluate_policy output
        "summary"     -- DataFrame with MeanReturn, StdReturn, etc.

    Used in
    -------
    main.ipynb Section 9 -- Trained Policy Evaluation.
    """
    q_sarsa = comparison_out["trained"]["q_sarsa"]
    q_q = comparison_out["trained"]["q_qlearning"]
    reinforce_nets = comparison_out["trained"]["reinforce_nets"]
    global_table = comparison_out["tables"]["global"]

    best_arch = choose_best_reinforce_arch(global_table)
    best_net = reinforce_nets[best_arch]

    # Build deterministic (greedy) policy functions from trained models
    def sarsa_policy(st: List[int]) -> int:
        return greedy_action_from_q(q_sarsa, st)

    def q_policy(st: List[int]) -> int:
        return greedy_action_from_q(q_q, st)

    def r_policy(st: List[int]) -> int:
        return greedy_action_from_net(best_net, st, config)

    eval_sarsa = evaluate_policy(sarsa_policy, config, num_episodes=num_eval_episodes)
    eval_q = evaluate_policy(q_policy, config, num_episodes=num_eval_episodes)
    eval_r = evaluate_policy(r_policy, config, num_episodes=num_eval_episodes)

    # Attach policy_fn and config so plot_financial_diagnostics can build heatmaps
    eval_sarsa["policy_fn"] = sarsa_policy
    eval_q["policy_fn"] = q_policy
    eval_r["policy_fn"] = r_policy
    eval_sarsa["config"] = config
    eval_q["config"] = config
    eval_r["config"] = config

    eval_dict: Dict[str, Dict[str, object]] = {
        "SARSA": eval_sarsa,
        "Q-learning": eval_q,
        f"REINFORCE-{best_arch}": eval_r,
    }

    if include_twap:
        def twap_fn(st: List[int]) -> int:
            return twap_policy(st, config)
        eval_twap = evaluate_policy(twap_fn, config, num_episodes=num_eval_episodes)
        eval_twap["policy_fn"] = twap_fn
        eval_twap["config"] = config
        eval_dict["TWAP"] = eval_twap

    summary_rows = []
    for name, out in eval_dict.items():
        summary_rows.append(
            {
                "Policy": name,
                "MeanReturn": float(out["mean_return"]),
                "StdReturn": float(out["std_return"]),
                "MeanFinalInventory": float(out["mean_inventory"][-1]),
                "MeanAction_t0": float(out["mean_action"][0]),
                "MeanAction_tLast": float(out["mean_action"][-1]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("MeanReturn", ascending=False)
    plot_financial_diagnostics(eval_dict, t_fixed_for_heatmap=0)

    return {
        "best_arch": best_arch,
        "evaluations": eval_dict,
        "summary": summary_df,
    }


# ===================================================================
# REINFORCE ablation (2x2 factorial)
# ===================================================================

def compare_reinforce_stability_2x2(
    config: Dict[str, object],
    num_episodes: int = 1000,
    num_runs: int = 6,
    K: int = 50,
    lr: float = 3e-4,
    arch: str = "medium",
    clip_value: float = 1.0,
    band: str = "std",
    smooth_window: int = 50,
    base_seed: int = 123,
    export_csv_prefix: str | None = None,
) -> Dict[str, object]:
    """2x2 factorial ablation: baseline (on/off) x gradient clipping (on/off).

    Trains REINFORCE under all four configurations with seeded runs for
    reproducibility, and produces:
    1. Raw learning curves with bands
    2. Smoothed learning curves
    3. Gradient-norm diagnostic plot (pre-clipping norms)
    4. Summary table with FinalMean, variance, and gradient statistics

    Parameters
    ----------
    base_seed : int
        Each (config, run) pair gets a deterministic seed derived from this.

    Returns
    -------
    dict with keys:
        "runs"    -- {label: ndarray(num_runs, num_episodes)}
        "grads"   -- {label: ndarray(num_runs, num_episodes)}
        "stats"   -- {label: (mean, std, stderr)}
        "summary" -- DataFrame

    Used in
    -------
    main.ipynb Section 6 -- REINFORCE Ablation Study.
    """
    configs = [
        (True, True, "baseline+clipping"),
        (True, False, "baseline+no_clip"),
        (False, True, "no_baseline+clipping"),
        (False, False, "no_baseline+no_clip"),
    ]

    runs_by_cfg: Dict[str, np.ndarray] = {}
    grads_by_cfg: Dict[str, np.ndarray] = {}
    stats_by_cfg: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for use_baseline, use_clipping, label in configs:
        run_curves = np.zeros((num_runs, num_episodes))
        grad_curves = np.zeros((num_runs, num_episodes))

        for r in tqdm(range(num_runs), desc=f"Ablation runs: {label}"):
            # Deterministic seed: unique per (config index, run index)
            set_all_seeds(base_seed + 1000 * len(runs_by_cfg) + r)
            _, returns, grad_norms = reinforce_with_options(
                config,
                num_episodes=num_episodes,
                K=K,
                lr=lr,
                arch=arch,
                use_baseline=use_baseline,
                use_clipping=use_clipping,
                clip_value=clip_value,
            )
            run_curves[r, :] = np.asarray(returns)
            grad_curves[r, :] = np.asarray(grad_norms)

        runs_by_cfg[label] = run_curves
        grads_by_cfg[label] = grad_curves
        stats_by_cfg[label] = summarize_runs(run_curves)

    x = np.arange(num_episodes)

    # --- Plot 1: raw learning curves ---
    plt.figure(figsize=(11, 5))
    for _, _, label in configs:
        mean_, std_, stderr_ = stats_by_cfg[label]
        plot_with_bands(x, mean_, std_, stderr_, label, band=band)
    plt.xlabel("Episodes / updates")
    plt.ylabel("Average return")
    plt.title(f"REINFORCE ablation (arch={arch}, band={band})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot 2: smoothed means ---
    if smooth_window > 1:
        plt.figure(figsize=(11, 5))
        x_s = x[smooth_window - 1 :]
        for _, _, label in configs:
            mean_, _, _ = stats_by_cfg[label]
            plt.plot(x_s, moving_average(mean_, smooth_window), label=f"{label} (MA {smooth_window})")
        plt.xlabel("Episodes / updates")
        plt.ylabel("Average return")
        plt.title("REINFORCE ablation (smoothed means)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Plot 3: gradient norm diagnostic ---
    plt.figure(figsize=(11, 5))
    for _, _, label in configs:
        g_mean = grads_by_cfg[label].mean(axis=0)
        plt.plot(x, g_mean, label=label)
    plt.xlabel("Episodes / updates")
    plt.ylabel("Mean grad norm (pre-clip)")
    plt.title("Gradient norm diagnostic by configuration")
    plt.legend()
    plt.tight_layout()
    plt.show()

    summary_rows = []
    for use_baseline, use_clipping, label in configs:
        mean_, std_, stderr_ = stats_by_cfg[label]
        final_idx = len(mean_) - 1
        best_idx = int(np.argmax(mean_))
        summary_rows.append(
            {
                "Config": label,
                "use_baseline": use_baseline,
                "use_clipping": use_clipping,
                "FinalMean": float(mean_[final_idx]),
                "FinalStdAcrossRuns": float(std_[final_idx]),
                "FinalStdErr": float(stderr_[final_idx]),
                "BestIdx": best_idx,
                "BestMean": float(mean_[best_idx]),
                "MeanGradNorm(pre-clip)": float(grads_by_cfg[label].mean()),
                "P95GradNorm(pre-clip)": float(np.percentile(grads_by_cfg[label], 95)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("FinalMean", ascending=False)
    if export_csv_prefix:
        summary_df.to_csv(f"{export_csv_prefix}_reinforce_ablation_summary.csv", index=False)

    return {"runs": runs_by_cfg, "grads": grads_by_cfg, "stats": stats_by_cfg, "summary": summary_df}


# ===================================================================
# Deep-dive diagnostics (main.ipynb Section 10)
# ===================================================================

def tabular_lr_sensitivity(
    config: Dict[str, object],
    lr_values: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    num_episodes: int = 1000,
    num_runs: int = 10,
    epoch_size: int = 25,
    smooth_window: int = 50,
) -> Dict[str, object]:
    """Sweep the learning rate (eta) for both SARSA and Q-learning.

    For each eta value, trains both algorithms over ``num_runs`` independent
    runs and plots mean convergence curves side-by-side.  This answers
    whether underperformance of a tabular method is due to a suboptimal
    learning rate or a structural limitation.

    Returns
    -------
    dict with keys "sarsa" and "qlearning", each mapping eta (float) to
    the mean learning curve (ndarray of length num_episodes).

    Used in
    -------
    main.ipynb Section 10.2 -- Learning Rate Sensitivity.
    """
    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])
    num_time_steps = int(config["NUM_TIME_STEPS"])
    actions = list(config["ACTIONS"])
    epsilon = float(config["epsilon"])

    results: Dict[str, Dict[float, np.ndarray]] = {"sarsa": {}, "qlearning": {}}

    for eta in lr_values:
        sarsa_runs = np.zeros((num_runs, num_episodes))
        q_runs = np.zeros((num_runs, num_episodes))

        for r in tqdm(range(num_runs), desc=f"LR sensitivity eta={eta:.2f}"):
            q_sarsa = np.zeros((num_blocks, num_s, num_time_steps, len(actions)))
            q_q = np.zeros_like(q_sarsa)
            for ep in range(num_episodes):
                eps = epsilon * ((1 - epsilon) ** (ep // epoch_size))
                sarsa_runs[r, ep] = sarsa(q_sarsa, config, eta=eta, eps=eps)
                q_runs[r, ep] = q_learning(q_q, config, eta=eta, eps=eps)

        results["sarsa"][eta] = sarsa_runs.mean(axis=0)
        results["qlearning"][eta] = q_runs.mean(axis=0)

    x = np.arange(num_episodes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for eta in lr_values:
        axes[0].plot(x, results["sarsa"][eta], label=f"eta={eta:.2f}")
        axes[1].plot(x, results["qlearning"][eta], label=f"eta={eta:.2f}")
    axes[0].set_title("SARSA -- learning-rate sensitivity")
    axes[1].set_title("Q-learning -- learning-rate sensitivity")
    for ax in axes:
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Mean return")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    if smooth_window > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x_s = x[smooth_window - 1 :]
        for eta in lr_values:
            axes[0].plot(x_s, moving_average(results["sarsa"][eta], smooth_window), label=f"eta={eta:.2f}")
            axes[1].plot(x_s, moving_average(results["qlearning"][eta], smooth_window), label=f"eta={eta:.2f}")
        axes[0].set_title(f"SARSA -- LR sensitivity (MA {smooth_window})")
        axes[1].set_title(f"Q-learning -- LR sensitivity (MA {smooth_window})")
        for ax in axes:
            ax.set_xlabel("Episodes")
            ax.set_ylabel("Mean return")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    return results


def plot_q_table_diagnostics(
    q_sarsa: np.ndarray,
    q_qlearning: np.ndarray,
    config: Dict[str, object],
    t_fixed: int = 0,
    all_timesteps: bool = False,
) -> Dict[str, object]:
    """Side-by-side diagnostics comparing two trained Q-tables (SARSA vs Q-learning).

    Produces either:
      - a 2x2 grid at one fixed timestep (default, backward compatible), or
      - a compact 4xT figure covering all timesteps when all_timesteps=True.

    For each plotted timestep:
      - Value landscape: V(x, s) = max_a Q(x, s, t, a)
      - Policy entropy: H(x, s) = -sum_a pi(a) log pi(a)
        where pi is the softmax of Q-values (higher = more uncertain)

    Also prints a sparsity report: what fraction of Q-table entries are
    non-zero.  High sparsity means many state-action pairs were never
    updated during training.

    Returns
    -------
    dict mapping algorithm name to {"nonzero_fraction", "total_entries"}.

    Used in
    -------
    main.ipynb Section 10.3 -- Q-Table Diagnostics.
    """
    tables = {"SARSA": q_sarsa, "Q-learning": q_qlearning}
    diagnostics: Dict[str, object] = {}

    for name, qtab in tables.items():
        nonzero_frac = float((qtab != 0).sum()) / max(qtab.size, 1)
        diagnostics[name] = {"nonzero_fraction": nonzero_frac, "total_entries": qtab.size}

    if all_timesteps:
        num_time_steps = int(config["NUM_TIME_STEPS"])
        fig, axes = plt.subplots(4, num_time_steps, figsize=(2.2 * num_time_steps, 12), squeeze=False)

        for t in range(num_time_steps):
            # SARSA value
            sarsa_slice = q_sarsa[:, :, t, :]
            sarsa_v = sarsa_slice.max(axis=-1)
            im = axes[0, t].imshow(sarsa_v, origin="lower", aspect="auto", cmap="viridis")
            axes[0, t].set_title(f"SARSA V (t={t})")
            if t == 0:
                axes[0, t].set_ylabel("Inventory x")
            axes[0, t].set_xlabel("Price state s")
            fig.colorbar(im, ax=axes[0, t], fraction=0.046, pad=0.04)

            # Q-learning value
            qlearn_slice = q_qlearning[:, :, t, :]
            qlearn_v = qlearn_slice.max(axis=-1)
            im = axes[1, t].imshow(qlearn_v, origin="lower", aspect="auto", cmap="viridis")
            axes[1, t].set_title(f"Q-learning V (t={t})")
            if t == 0:
                axes[1, t].set_ylabel("Inventory x")
            axes[1, t].set_xlabel("Price state s")
            fig.colorbar(im, ax=axes[1, t], fraction=0.046, pad=0.04)

            # SARSA entropy
            sarsa_max = sarsa_slice.max(axis=-1, keepdims=True)
            sarsa_exp = np.exp(sarsa_slice - sarsa_max)  # stable softmax
            sarsa_probs = sarsa_exp / (sarsa_exp.sum(axis=-1, keepdims=True) + 1e-12)
            sarsa_entropy = -(sarsa_probs * np.log(sarsa_probs + 1e-12)).sum(axis=-1)
            im = axes[2, t].imshow(sarsa_entropy, origin="lower", aspect="auto", cmap="inferno")
            axes[2, t].set_title(f"SARSA entropy (t={t})")
            if t == 0:
                axes[2, t].set_ylabel("Inventory x")
            axes[2, t].set_xlabel("Price state s")
            fig.colorbar(im, ax=axes[2, t], fraction=0.046, pad=0.04)

            # Q-learning entropy
            qlearn_max = qlearn_slice.max(axis=-1, keepdims=True)
            qlearn_exp = np.exp(qlearn_slice - qlearn_max)  # stable softmax
            qlearn_probs = qlearn_exp / (qlearn_exp.sum(axis=-1, keepdims=True) + 1e-12)
            qlearn_entropy = -(qlearn_probs * np.log(qlearn_probs + 1e-12)).sum(axis=-1)
            im = axes[3, t].imshow(qlearn_entropy, origin="lower", aspect="auto", cmap="inferno")
            axes[3, t].set_title(f"Q-learning entropy (t={t})")
            if t == 0:
                axes[3, t].set_ylabel("Inventory x")
            axes[3, t].set_xlabel("Price state s")
            fig.colorbar(im, ax=axes[3, t], fraction=0.046, pad=0.04)

        plt.suptitle("Q-table diagnostics (all timesteps)", y=1.02, fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for col, (name, qtab) in enumerate(tables.items()):
            # Value landscape: best achievable value at each (x, s) for fixed t
            v = qtab[:, :, t_fixed, :].max(axis=-1)
            im0 = axes[0, col].imshow(v, origin="lower", aspect="auto", cmap="viridis")
            axes[0, col].set_title(f"{name} -- V(x, s, t={t_fixed})")
            axes[0, col].set_xlabel("Price state s")
            axes[0, col].set_ylabel("Inventory x")
            fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

            # Policy entropy via softmax of Q-values (numerically stable)
            q_slice = qtab[:, :, t_fixed, :]
            q_max = q_slice.max(axis=-1, keepdims=True)
            exp_q = np.exp(q_slice - q_max)  # subtract max for numerical stability
            probs = exp_q / (exp_q.sum(axis=-1, keepdims=True) + 1e-12)
            entropy = -(probs * np.log(probs + 1e-12)).sum(axis=-1)
            im1 = axes[1, col].imshow(entropy, origin="lower", aspect="auto", cmap="inferno")
            axes[1, col].set_title(f"{name} -- policy entropy (t={t_fixed})")
            axes[1, col].set_xlabel("Price state s")
            axes[1, col].set_ylabel("Inventory x")
            fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        plt.suptitle("Q-table diagnostics", y=1.02, fontsize=14)
        plt.tight_layout()
        plt.show()

    print("Sparsity report:")
    for name, d in diagnostics.items():
        print(f"  {name}: {d['nonzero_fraction']:.2%} of {d['total_entries']} entries are non-zero")

    return diagnostics


def estimate_state_visitation(
    config: Dict[str, object],
    num_episodes: int = 5000,
    t_fixed: int = 0,
) -> Dict[str, np.ndarray]:
    """Estimate which states each tabular method actually visits during training.

    Runs ``num_episodes`` of SARSA and Q-learning from scratch (fresh
    Q-tables), counting visits N(x, s, t) at every step.  Plots side-by-side
    heatmaps at ``t=t_fixed`` and prints the fraction of the total state
    space (N x M x T = 1200 states) that was visited at least once.

    This directly tests the hypothesis that tabular methods underperform
    because large regions of the state space are never explored.

    Returns
    -------
    {"sarsa": visit_counts, "qlearning": visit_counts}
    Each is an ndarray of shape (NUM_BLOCKS, NUM_S, NUM_TIME_STEPS).

    Used in
    -------
    main.ipynb Section 10.4 -- State Visitation Analysis.
    """
    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])
    num_time_steps = int(config["NUM_TIME_STEPS"])
    actions = list(config["ACTIONS"])
    epsilon = float(config["epsilon"])

    visit_sarsa = np.zeros((num_blocks, num_s, num_time_steps))
    visit_q = np.zeros((num_blocks, num_s, num_time_steps))
    q_sarsa = np.zeros((num_blocks, num_s, num_time_steps, len(actions)))
    q_q = np.zeros_like(q_sarsa)

    start = [int(config["NUM_BLOCKS"]) - 1, int(config["S0"]), 0]

    # --- SARSA visitation (on-policy: action chosen before stepping) ---
    for ep in tqdm(range(num_episodes), desc="State visitation (SARSA)"):
        eps = epsilon * ((1 - epsilon) ** (ep // 25))
        state = start.copy()
        action = epsilon_greedy(state, q_sarsa, config, eps)
        while state[2] < num_time_steps - 1:
            x, s, t = state[0], state[1], state[2]
            if 0 <= x < num_blocks and 0 <= s < num_s and 0 <= t < num_time_steps:
                visit_sarsa[x, s, t] += 1
            next_state, reward = transition(state, action, config)
            next_action = epsilon_greedy(next_state, q_sarsa, config, eps)
            # On-policy TD update (same as sarsa() in algorithms.py)
            q_cur = q_sarsa[state[0], state[1], state[2], action]
            q_nxt = q_sarsa[next_state[0], next_state[1], next_state[2], next_action]
            q_sarsa[state[0], state[1], state[2], action] += float(config["eta"]) * (reward + q_nxt - q_cur)
            state = next_state
            action = next_action

    # --- Q-learning visitation (off-policy: max over next actions) ---
    for ep in tqdm(range(num_episodes), desc="State visitation (Q-learning)"):
        eps = epsilon * ((1 - epsilon) ** (ep // 25))
        state = start.copy()
        while state[2] < num_time_steps - 1:
            x, s, t = state[0], state[1], state[2]
            if 0 <= x < num_blocks and 0 <= s < num_s and 0 <= t < num_time_steps:
                visit_q[x, s, t] += 1
            action = epsilon_greedy(state, q_q, config, eps)
            next_state, reward = transition(state, action, config)
            # Off-policy TD update (same as q_learning() in algorithms.py)
            q_cur = q_q[state[0], state[1], state[2], action]
            q_max = np.max(q_q[next_state[0], next_state[1], next_state[2], :])
            q_q[state[0], state[1], state[2], action] += float(config["eta"]) * (reward + q_max - q_cur)
            state = next_state

    total_states = num_blocks * num_s * num_time_steps

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, visits) in zip(axes, [("SARSA", visit_sarsa), ("Q-learning", visit_q)]):
        hmap = visits[:, :, t_fixed]
        im = ax.imshow(hmap, origin="lower", aspect="auto", cmap="hot")
        ax.set_title(f"{name} -- visits at t={t_fixed}")
        ax.set_xlabel("Price state s")
        ax.set_ylabel("Inventory x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("State visitation heatmaps", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

    for name, visits in [("SARSA", visit_sarsa), ("Q-learning", visit_q)]:
        visited = int((visits > 0).sum())
        print(f"  {name}: visited {visited}/{total_states} states "
              f"({visited / total_states:.1%})")

    return {"sarsa": visit_sarsa, "qlearning": visit_q}


def plot_policy_state_space(
    net: torch.nn.Module,
    config: Dict[str, object],
    q_sarsa: np.ndarray,
    q_qlearn: np.ndarray,
    time_steps: Sequence[int] = (0, 4, 8),
    value_rollouts: int = 100,
) -> None:
    """Visualize policy behavior directly in the discrete state space.

    Unlike parameter-space landscapes, this diagnostic works in the actual MDP
    state variables and exploits the small grid size to evaluate every state.
    It generates four figures:

    1. REINFORCE value surfaces ``V_pi(t, x, s)`` (3-D + heatmap) for selected times
    2. Action-probability heatmaps ``pi(a|t,x,s)`` for each action across selected times
    3. Policy entropy surfaces/heatmaps across selected times
    4. Greedy-policy comparison across selected times
       (REINFORCE vs SARSA vs Q-learning + disagreement map)

    Parameters
    ----------
    net : MLPPolicyNet
        Trained REINFORCE policy network.
    config : dict
        Environment/training configuration dictionary.
    q_sarsa, q_qlearn : ndarray
        Trained tabular Q-values of shape (NUM_BLOCKS, NUM_S, NUM_TIME_STEPS, |A|).
    time_steps : sequence of int
        Time slices at which to visualize all state-space objects.
    value_rollouts : int
        Monte Carlo rollouts per state used to estimate ``V_pi``.

    Used in
    -------
    main.ipynb Section 10.5 -- State-space policy deep dive.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3-D projection

    num_blocks = int(config["NUM_BLOCKS"])
    num_s = int(config["NUM_S"])
    num_time_steps = int(config["NUM_TIME_STEPS"])
    actions = list(config["ACTIONS"])
    device = config["DEVICE"]  # type: ignore[assignment]
    eps = 1e-12

    if len(time_steps) == 0:
        raise ValueError("time_steps must contain at least one time index")

    clean_time_steps = sorted({int(np.clip(t, 0, num_time_steps - 1)) for t in time_steps})

    def _policy_probs(state: Sequence[int]) -> np.ndarray:
        """Compute masked softmax probabilities pi(.|state) from the network."""
        x, s, t = int(state[0]), int(state[1]), int(state[2])
        inp = make_input(t, x, s, config).unsqueeze(0)
        actions_tensor = torch.tensor(actions, device=device)
        with torch.no_grad():
            logits = net(inp).squeeze(0)
            valid_mask = actions_tensor <= x
            masked_logits = logits.masked_fill(~valid_mask, -1e9)
            probs = torch.softmax(masked_logits, dim=0).detach().cpu().numpy()
        return probs

    def _rollout_value_from_state(x0: int, s0: int, t0: int) -> float:
        """Estimate V_pi(t0, x0, s0) by Monte Carlo trajectories."""
        ret = 0.0
        for _ in range(value_rollouts):
            state = [x0, s0, t0]
            g = 0.0
            while state[2] < num_time_steps - 1:
                probs = _policy_probs(state)
                action_idx = int(np.random.choice(len(actions), p=probs))
                action = int(actions[action_idx])
                action = min(action, state[0])
                next_state, reward = transition(state, action, config)
                g += reward
                state = next_state
            ret += g
        return ret / value_rollouts

    inv_grid = np.arange(num_blocks)
    price_grid = np.arange(num_s)
    X, S = np.meshgrid(inv_grid, price_grid, indexing="ij")

    # -------------------------------------------------------------------
    # Figure 1: V_pi surfaces and heatmaps across selected time slices
    # -------------------------------------------------------------------
    n_t = len(clean_time_steps)
    fig1 = plt.figure(figsize=(14, 5 * n_t))

    for row, t_fixed in enumerate(clean_time_steps):
        vmap = np.zeros((num_blocks, num_s))
        for x in range(num_blocks):
            for s in range(num_s):
                vmap[x, s] = _rollout_value_from_state(x, s, t_fixed)

        ax3d = fig1.add_subplot(n_t, 2, 2 * row + 1, projection="3d")
        ax3d.plot_surface(X, S, vmap, cmap="viridis", edgecolor="none", alpha=0.95)
        ax3d.set_title(f"V_pi surface at t={t_fixed}")
        ax3d.set_xlabel("Inventory x")
        ax3d.set_ylabel("Price state s")
        ax3d.set_zlabel("Expected return")
        ax3d.view_init(elev=28, azim=-55)

        ax2d = fig1.add_subplot(n_t, 2, 2 * row + 2)
        im = ax2d.imshow(vmap, origin="lower", aspect="auto", cmap="viridis")
        ax2d.set_title(f"V_pi heatmap at t={t_fixed}")
        ax2d.set_xlabel("Price state s")
        ax2d.set_ylabel("Inventory x")
        fig1.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)

    plt.suptitle("REINFORCE value landscape in state space", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------
    # Figure 2: Action-probability landscape across selected time slices
    # -------------------------------------------------------------------
    fig2, axes2 = plt.subplots(n_t, len(actions), figsize=(3.6 * len(actions), 3.0 * n_t), squeeze=False)
    prob_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for row, t_fixed in enumerate(clean_time_steps):
        probs_per_action = np.zeros((len(actions), num_blocks, num_s))
        for x in range(num_blocks):
            for s in range(num_s):
                probs = _policy_probs([x, s, t_fixed])
                probs_per_action[:, x, s] = probs

        for action_idx, action in enumerate(actions):
            ax = axes2[row, action_idx]
            im = ax.imshow(
                probs_per_action[action_idx],
                origin="lower",
                aspect="auto",
                cmap="magma",
                norm=prob_norm,
            )
            ax.set_title(f"P(a={action} | t={t_fixed}, x, s)")
            ax.set_xlabel("Price state s")
            ax.set_ylabel("Inventory x")
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("REINFORCE action-probability landscape across selected times", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------
    # Figure 3: Entropy surfaces and heatmaps
    # -------------------------------------------------------------------
    fig3 = plt.figure(figsize=(14, 5 * n_t))
    for row, t_fixed in enumerate(clean_time_steps):
        hmap = np.zeros((num_blocks, num_s))
        for x in range(num_blocks):
            for s in range(num_s):
                probs = _policy_probs([x, s, t_fixed])
                hmap[x, s] = -float(np.sum(probs * np.log(probs + eps)))

        ax3d = fig3.add_subplot(n_t, 2, 2 * row + 1, projection="3d")
        ax3d.plot_surface(X, S, hmap, cmap="plasma", edgecolor="none", alpha=0.95)
        ax3d.set_title(f"Policy entropy at t={t_fixed}")
        ax3d.set_xlabel("Inventory x")
        ax3d.set_ylabel("Price state s")
        ax3d.set_zlabel("Entropy")
        ax3d.view_init(elev=28, azim=-55)

        ax2d = fig3.add_subplot(n_t, 2, 2 * row + 2)
        im = ax2d.imshow(hmap, origin="lower", aspect="auto", cmap="plasma")
        ax2d.set_title(f"Entropy heatmap at t={t_fixed}")
        ax2d.set_xlabel("Price state s")
        ax2d.set_ylabel("Inventory x")
        fig3.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)

    plt.suptitle("REINFORCE policy uncertainty in state space", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------
    # Figure 4: Cross-algorithm greedy policy comparison across time slices
    # -------------------------------------------------------------------
    reinforce_greedy = np.zeros((n_t, num_blocks, num_s), dtype=int)
    sarsa_greedy = np.zeros((n_t, num_blocks, num_s), dtype=int)
    qlearning_greedy = np.zeros((n_t, num_blocks, num_s), dtype=int)
    disagreement = np.zeros((n_t, num_blocks, num_s), dtype=int)
    action_to_idx = {int(a): idx for idx, a in enumerate(actions)}

    for t_idx, t_fixed in enumerate(clean_time_steps):
        for x in range(num_blocks):
            for s in range(num_s):
                probs = _policy_probs([x, s, t_fixed])
                r_action = int(actions[int(np.argmax(probs))])
                r_action = min(r_action, x)
                n_valid = min(x + 1, len(actions))
                s_action_idx = int(np.argmax(q_sarsa[x, s, t_fixed, :n_valid]))
                q_action_idx = int(np.argmax(q_qlearn[x, s, t_fixed, :n_valid]))
                s_action = int(actions[s_action_idx])
                q_action = int(actions[q_action_idx])

                reinforce_greedy[t_idx, x, s] = action_to_idx[r_action]
                sarsa_greedy[t_idx, x, s] = action_to_idx[s_action]
                qlearning_greedy[t_idx, x, s] = action_to_idx[q_action]
                disagreement[t_idx, x, s] = len({r_action, s_action, q_action}) - 1

    action_cmap = plt.get_cmap("viridis", len(actions))
    action_norm = mcolors.BoundaryNorm(
        boundaries=np.arange(-0.5, len(actions) + 0.5, 1.0),
        ncolors=action_cmap.N,
    )
    disagree_cmap = plt.get_cmap("inferno", 3)
    disagree_norm = mcolors.BoundaryNorm(
        boundaries=np.array([-0.5, 0.5, 1.5, 2.5]),
        ncolors=disagree_cmap.N,
    )

    fig4, axes4 = plt.subplots(n_t, 4, figsize=(18, 3.8 * n_t), squeeze=False)
    plots = [
        ("REINFORCE greedy", reinforce_greedy, "action"),
        ("SARSA greedy", sarsa_greedy, "action"),
        ("Q-learning greedy", qlearning_greedy, "action"),
        ("Disagreement count", disagreement, "disagreement"),
    ]
    for row, t_fixed in enumerate(clean_time_steps):
        for col, (title, mats, kind) in enumerate(plots):
            ax = axes4[row, col]
            mat = mats[row]
            if kind == "action":
                im = ax.imshow(mat, origin="lower", aspect="auto", cmap=action_cmap, norm=action_norm)
            else:
                im = ax.imshow(mat, origin="lower", aspect="auto", cmap=disagree_cmap, norm=disagree_norm)
            ax.set_title(f"{title} (t={t_fixed})")
            ax.set_xlabel("Price state s")
            ax.set_ylabel("Inventory x")
            cbar = fig4.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if kind == "action":
                cbar.set_ticks(np.arange(len(actions)))
                cbar.set_ticklabels([str(a) for a in actions])
                cbar.set_label("Action taken")
            else:
                cbar.set_ticks([0, 1, 2])
                cbar.set_label("Disagreement count")

    plt.suptitle("Policy comparison in state space across selected times", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_reinforce_loss_landscape(
    net: torch.nn.Module,
    config: Dict[str, object],
    K: int = 50,
    num_points: int = 21,
    scale: float = 1.0,
    num_eval: int = 5,
) -> None:
    """Visualize the expected-return landscape around a trained REINFORCE policy.

    Samples two random **orthogonal** directions in parameter space (via
    Gram-Schmidt) and evaluates the average trajectory return on a 2-D grid
    of perturbations theta* + alpha*d1 + beta*d2.

    Produces three panels:
    1. 2-D filled contour plot of the return landscape (star marks trained params)
    2. 3-D surface plot of the same landscape for intuitive curvature reading
    3. 1-D cross-section along direction 1

    A smooth, well-conditioned landscape with a clear peak at theta*
    indicates that gradient-based optimization can reliably find good
    policies -- an advantage over tabular methods whose entries are updated
    independently.

    Parameters
    ----------
    net : MLPPolicyNet
        Trained policy (parameters are temporarily perturbed and restored).
    K : int
        Trajectories per evaluation point.
    num_points : int
        Grid resolution along each direction.
    scale : float
        Range of perturbation: alpha, beta in [-scale, +scale].
    num_eval : int
        Number of independent evaluations averaged at each grid point
        (reduces stochastic noise).

    Used in
    -------
    main.ipynb Section 10.5 -- REINFORCE Loss Landscape.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3-D projection

    device = next(net.parameters()).device
    # Flatten all parameters into a single 1-D vector
    params_flat = torch.cat([p.detach().flatten() for p in net.parameters()])
    n_params = params_flat.numel()

    # Sample two random orthogonal directions via Gram-Schmidt
    d1 = torch.randn(n_params, device=device)
    d1 = d1 / d1.norm()
    d2 = torch.randn(n_params, device=device)
    d2 = d2 - d2.dot(d1) * d1  # remove component along d1
    d2 = d2 / d2.norm()

    alphas = np.linspace(-scale, scale, num_points)
    betas = np.linspace(-scale, scale, num_points)

    landscape = np.zeros((num_points, num_points))
    cross_section = np.zeros(num_points)

    shapes = [p.shape for p in net.parameters()]
    sizes = [p.numel() for p in net.parameters()]

    def _set_params(flat: torch.Tensor) -> None:
        """Copy a flat parameter vector back into the network's parameters."""
        offset = 0
        for p, sz, sh in zip(net.parameters(), sizes, shapes):
            p.data.copy_(flat[offset : offset + sz].view(sh))
            offset += sz

    # --- Evaluate on the 2-D grid ---
    for i, alpha in enumerate(tqdm(alphas, desc="Loss landscape rows")):
        for j, beta in enumerate(betas):
            new_flat = params_flat + alpha * d1 + beta * d2
            _set_params(new_flat)
            returns_sum = 0.0
            for _ in range(num_eval):
                with torch.no_grad():
                    _, batch_r = generate_trajectories_batch(net, K, config)
                returns_sum += float(batch_r.mean().item())
            landscape[i, j] = returns_sum / num_eval

    # --- Evaluate on the 1-D cross-section (beta=0) ---
    for i, alpha in enumerate(alphas):
        new_flat = params_flat + alpha * d1
        _set_params(new_flat)
        returns_sum = 0.0
        for _ in range(num_eval):
            with torch.no_grad():
                _, batch_r = generate_trajectories_batch(net, K, config)
            returns_sum += float(batch_r.mean().item())
        cross_section[i] = returns_sum / num_eval

    # Restore original parameters
    _set_params(params_flat)

    A, B = np.meshgrid(alphas, betas, indexing="ij")

    # --- Panel layout: contour + 3-D surface on top row, cross-section below ---
    fig = plt.figure(figsize=(16, 10))

    # Panel 1: 2-D contour
    ax_contour = fig.add_subplot(2, 2, 1)
    cf = ax_contour.contourf(A, B, landscape, levels=30, cmap="RdYlGn")
    ax_contour.set_xlabel("Direction 1")
    ax_contour.set_ylabel("Direction 2")
    ax_contour.set_title("Return landscape (2-D contour)")
    ax_contour.plot(0, 0, "k*", markersize=12)
    fig.colorbar(cf, ax=ax_contour, fraction=0.046, pad=0.04)

    # Panel 2: 3-D surface
    ax_3d = fig.add_subplot(2, 2, 2, projection="3d")
    ax_3d.plot_surface(
        A, B, landscape, cmap="RdYlGn", edgecolor="none", alpha=0.9,
    )
    ax_3d.set_xlabel("Direction 1")
    ax_3d.set_ylabel("Direction 2")
    ax_3d.set_zlabel("Expected return")
    ax_3d.set_title("Return landscape (3-D surface)")
    # Mark the trained optimum on the surface
    center_val = landscape[num_points // 2, num_points // 2]
    ax_3d.scatter([0], [0], [center_val], color="black", s=80, marker="*", zorder=5)
    ax_3d.view_init(elev=30, azim=-60)

    # Panel 3: 1-D cross-section (spans the bottom row)
    ax_cross = fig.add_subplot(2, 1, 2)
    ax_cross.plot(alphas, cross_section, "b-", linewidth=2)
    ax_cross.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax_cross.set_xlabel("Displacement along direction 1")
    ax_cross.set_ylabel("Expected return")
    ax_cross.set_title("Return cross-section (1-D)")

    plt.suptitle("REINFORCE policy -- return landscape", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()
