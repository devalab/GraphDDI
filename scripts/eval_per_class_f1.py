"""Run trainer.test on a checkpoint and dump per-class F1 as JSON.

Usage:
    uv run python scripts/eval_per_class_f1.py \\
        --config configs/experiment/drugbank_multiclass.yaml \\
        --ckpt path/to/last.ckpt --out outputs/eval/graphddi.json
"""

import argparse
import json
from pathlib import Path

import torch
from torchmetrics.classification import MulticlassF1Score

from graphddi.training.cli import GraphDDILightningCLI


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cli = GraphDDILightningCLI(
        args=["--config", args.config],
        run=False,
        save_config_callback=None,
    )
    model = cli.model.__class__.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()
    cli.datamodule.setup()
    loader = cli.datamodule.test_dataloader()

    metric = MulticlassF1Score(num_classes=model.hparams.num_classes, average=None)
    with torch.no_grad():
        for batch_a, batch_b, y in loader:
            logits = model(batch_a, batch_b)
            preds = logits.argmax(-1)
            metric.update(preds, y.long())

    per_class = metric.compute().tolist()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(per_class, f)
    print(f"Macro-F1: {sum(per_class) / len(per_class):.4f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
