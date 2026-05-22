"""Log-frequency plot of DDI event types in DrugBank."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_OUT = Path("outputs/figures/data_dist.pdf")


def plot_data_dist(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = df["Y"].astype(int).value_counts().sort_index()
    classes = np.arange(1, 87)
    freq = np.array([counts.get(c, 0) for c in classes], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(classes, freq, color="#3a7ebf")
    ax.set_yscale("log")
    ax.set_xlabel("DDI Type")
    ax.set_ylabel("Frequency (log)")
    ax.set_xlim(0, 87)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--mini",
        action="store_true",
        help="Use tests/fixtures/drugbank_mini.csv instead of full TDC DrugBank.",
    )
    args = p.parse_args()

    if args.mini:
        df = pd.read_csv("tests/fixtures/drugbank_mini.csv")
    else:
        from tdc.multi_pred import DDI

        from graphddi.data._loader import quiet_stderr

        with quiet_stderr():
            df = DDI(name="DrugBank", path=args.data_root).get_data()

    plot_data_dist(df, Path(args.out))


if __name__ == "__main__":
    main()
