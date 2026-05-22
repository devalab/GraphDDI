"""Inference CLI entry point.

Usage:
    uv run python scripts/predict.py pair  <SMILES1> <SMILES2> --task multiclass
    uv run python scripts/predict.py batch pairs.csv --task binary --out preds.csv
"""

from graphddi.inference.predict import app

if __name__ == "__main__":
    app()
