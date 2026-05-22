"""Run an Optuna hyperparameter sweep.

Usage:
    uv run python scripts/sweep.py --config configs/sweep/drugbank_mini_optuna.yaml
"""

from graphddi.sweeps.optuna_runner import main

if __name__ == "__main__":
    main()
