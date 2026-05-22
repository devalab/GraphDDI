"""One-shot script to materialise ``tests/fixtures/drugbank_mini.csv``.

Downloads DrugBank via TDC, samples ~200 interaction pairs covering ~30 unique
drugs across ≥10 of the 86 DDI types, and writes them as a small CSV that lives
in version control (the rest of ``data/`` is gitignored).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "drugbank_mini.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--min-classes", type=int, default=10)
    parser.add_argument("--max-drugs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from tdc.multi_pred import DDI
    except ImportError:
        print("ERROR: install tdc first: `uv add pytdc`", file=sys.stderr)
        return 2

    print(f"Downloading DrugBank via TDC into {args.data_root} …")
    ddi = DDI(name="DrugBank", path=args.data_root)
    df: pd.DataFrame = ddi.get_data()

    # Drop rows whose SMILES RDKit can't parse — same filter as the real datamodule.
    def parses(smi: str) -> bool:
        return Chem.MolFromSmiles(smi) is not None

    valid_mask = df["Drug1"].map(parses) & df["Drug2"].map(parses)
    df = df[valid_mask].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)

    # Pick a small set of drugs first, then keep only pairs involving them.
    drug_ids = pd.concat([df["Drug1_ID"], df["Drug2_ID"]]).unique()
    keep_drugs = set(rng.choice(drug_ids, size=min(args.max_drugs, len(drug_ids)), replace=False))

    mask = df["Drug1_ID"].isin(keep_drugs) & df["Drug2_ID"].isin(keep_drugs)
    candidate = df[mask].reset_index(drop=True)
    if len(candidate) < args.n_pairs:
        # Fall back to single-end membership if too few pairs.
        mask = df["Drug1_ID"].isin(keep_drugs) | df["Drug2_ID"].isin(keep_drugs)
        candidate = df[mask].reset_index(drop=True)

    by_class: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(candidate["Y"]):
        by_class[int(lab)].append(i)

    chosen_classes = sorted(by_class.keys())
    rng.shuffle(chosen_classes)

    sampled_idx: list[int] = []
    classes_used: set[int] = set()
    # First pass: take ≥1 example from each available class until we hit min_classes.
    for cls in chosen_classes:
        if len(classes_used) >= args.min_classes and len(sampled_idx) >= args.n_pairs // 2:
            break
        pick = rng.choice(by_class[cls], size=1)[0]
        sampled_idx.append(int(pick))
        classes_used.add(cls)

    # Second pass: top up to n_pairs with random additional rows.
    remaining = [i for i in range(len(candidate)) if i not in set(sampled_idx)]
    if remaining:
        topup_n = max(0, args.n_pairs - len(sampled_idx))
        topup = rng.choice(remaining, size=min(topup_n, len(remaining)), replace=False)
        sampled_idx.extend(int(t) for t in topup)

    fixture = candidate.iloc[sampled_idx].reset_index(drop=True)
    print(
        f"Sampled {len(fixture)} pairs across "
        f"{fixture['Y'].nunique()} DDI classes and "
        f"{pd.concat([fixture['Drug1_ID'], fixture['Drug2_ID']]).nunique()} unique drugs."
    )

    counts = Counter(fixture["Y"])
    print("Class distribution:", dict(counts.most_common()))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
