"""Market impact RL package -- optimal execution via reinforcement learning.

This package implements the full pipeline for training and evaluating RL agents
on a discrete-time market-impact MDP where a broker liquidates a block of
shares under price impact and inventory risk.

Modules
-------
environment
    Core MDP dynamics: one-step ``transition`` and ``epsilon_greedy`` action
    selection.  Every other module depends on this.
algorithms
    Training algorithms (``sarsa``, ``q_learning``, ``reinforce``,
    ``reinforce_with_options``) and the neural-network policy class
    (``MLPPolicyNet`` / ``build_policy_net``).
analysis
    Comparison protocols, statistical aggregation, post-training policy
    evaluation, and diagnostic utilities used in the notebook experiments.

Usage
-----
All public symbols are re-exported here so the notebook can do::

    from market_impact import compare_all_architectures, run_post_training_evaluation

Every function that touches MDP parameters accepts a ``config`` dict defined
once in the notebook (see main.ipynb Section 2).
"""

from .environment import epsilon_greedy, transition
from .algorithms import (
    MLPPolicyNet,
    build_policy_net,
    count_trainable_params,
    generate_trajectories_batch,
    make_input,
    q_learning,
    reinforce,
    reinforce_with_options,
    sarsa,
    set_all_seeds,
)
from .analysis import (
    choose_best_reinforce_arch,
    compare_all,
    compare_all_architectures,
    compare_fixed_trajectories_with_tables,
    compare_fixed_updates_with_tables,
    compare_reinforce_stability_2x2,
    compare_sarsa_qlearning,
    estimate_state_visitation,
    evaluate_policy,
    global_metrics,
    greedy_action_from_net,
    greedy_action_from_q,
    make_policy_heatmap,
    make_summary_table,
    moving_average,
    plot_financial_diagnostics,
    plot_policy_state_space,
    plot_q_table_diagnostics,
    plot_reinforce_loss_landscape,
    plot_with_bands,
    run_post_training_evaluation,
    summarize_runs,
    tabular_lr_sensitivity,
    twap_policy,
)

