# RL for Optimal Trade Execution

Reinforcement learning approaches to the market-impact problem: a broker must liquidate a position over discrete time steps while minimizing price impact and execution cost.

## Structure

- `main.ipynb` -- full pipeline: problem setup, training, evaluation, and diagnostics
- `market_impact/` -- Python package
  - `environment.py` -- MDP transition dynamics and epsilon-greedy action selection
  - `algorithms.py` -- SARSA, Q-learning, and REINFORCE (policy gradient)
  - `analysis.py` -- comparison protocols, plotting, and post-training evaluation

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install numpy torch tqdm matplotlib
```
