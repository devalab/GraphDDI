"""Per-class F1 polar plot — 2x2 grid for up to four models."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUT = Path("outputs/figures/f1_per_class_polar.pdf")


def _polar(ax, f1: np.ndarray, title: str) -> None:
    n = len(f1)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.bar(angles, f1, width=2 * np.pi / n, alpha=0.85, color="#3a7ebf")
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[::8])
    ax.set_xticklabels([str(i + 1) for i in range(0, n, 8)], fontsize=7)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7)
    ax.set_title(title, fontsize=10)


def plot_f1_polar(per_model_f1: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), subplot_kw=dict(projection="polar"))
    for ax, (name, f1) in zip(axes.flat, per_model_f1.items(), strict=False):
        _polar(ax, f1, name)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--per-class-f1",
        nargs="+",
        required=True,
        help='label=path pairs, e.g. "GraphDDI=outputs/eval/graphddi.json"',
    )
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    data: dict[str, np.ndarray] = {}
    for spec in args.per_class_f1:
        label, path = spec.split("=", 1)
        with open(path) as f:
            data[label] = np.asarray(json.load(f), dtype=float)
    plot_f1_polar(data, Path(args.out))


if __name__ == "__main__":
    main()
