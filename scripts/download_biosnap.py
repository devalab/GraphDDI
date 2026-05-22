"""Download BioSNAP DDI splits from the CASTER repository.

We use Huang et al.'s (CASTER) preprocessing of BioSNAP. The CASTER repo
bundles ``sup_train_val.csv`` (train+val combined) and ``sup_test.csv``.
This script downloads both and splits ``sup_train_val`` into train/val at a
7:1 ratio so the overall split is 70/10/20.

Resulting files written under ``<data-root>/biosnap/``::

    train.csv  val.csv  test.csv

Each CSV has columns ``SMILES1, SMILES2, Label``.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "biosnap"

SOURCES = {
    "sup_train_val.csv": (
        "https://raw.githubusercontent.com/kexinhuang12345/CASTER/master/"
        "DDE/data/BIOSNAP/sup_train_val.csv"
    ),
    "sup_test.csv": (
        "https://raw.githubusercontent.com/kexinhuang12345/CASTER/master/"
        "DDE/data/BIOSNAP/sup_test.csv"
    ),
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip (already present): {dest}")
        return
    print(f"  fetching {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CASTER columns to ``SMILES1, SMILES2, Label``."""
    rename = {
        "Drug1_SMILES": "SMILES1",
        "Drug2_SMILES": "SMILES2",
        "label": "Label",
    }
    df = df.rename(columns=rename)
    # Drop the unnamed CASTER index column if present.
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")
    # Cast label to int.
    df["Label"] = df["Label"].astype(float).round().astype(int)
    return df[["SMILES1", "SMILES2", "Label"]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--val-frac", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading BioSNAP CASTER splits …")
    raw = out / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        _download(url, raw / name)

    train_val = _normalise(pd.read_csv(raw / "sup_train_val.csv"))
    test = _normalise(pd.read_csv(raw / "sup_test.csv"))

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_val))
    n_val = int(args.val_frac * len(train_val))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_df = train_val.iloc[train_idx].reset_index(drop=True)
    val_df = train_val.iloc[val_idx].reset_index(drop=True)

    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    total = len(train_df) + len(val_df) + len(test)
    print(
        f"Wrote train={len(train_df)} val={len(val_df)} test={len(test)} "
        f"({len(train_df) / total:.0%}/{len(val_df) / total:.0%}/{len(test) / total:.0%})"
    )
    print(f"Files: {out}/train.csv,val.csv,test.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
