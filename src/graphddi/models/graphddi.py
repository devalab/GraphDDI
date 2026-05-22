"""GraphDDI Lightning module — encoder + interaction map + readout + head."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from graphddi.models.interaction import interaction_map

Task = Literal["binary", "multiclass"]


def _build_metrics(task: Task, num_classes: int) -> tuple[MetricCollection, str]:
    """Return the per-stage metric collection and the key tracked by ``MaxMetric``."""
    if task == "binary":
        return (
            MetricCollection(
                {
                    "acc": BinaryAccuracy(),
                    "precision": BinaryPrecision(),
                    "recall": BinaryRecall(),
                    "f1": BinaryF1Score(),
                    "auroc": BinaryAUROC(),
                    "pr_auc": BinaryAveragePrecision(),
                }
            ),
            "f1",
        )
    return (
        MetricCollection(
            {
                "precision_micro": MulticlassPrecision(num_classes=num_classes, average="micro"),
                "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
                "recall_micro": MulticlassRecall(num_classes=num_classes, average="micro"),
                "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
                "f1_micro": MulticlassF1Score(num_classes=num_classes, average="micro"),
                "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
                "auroc": MulticlassAUROC(num_classes=num_classes, average="macro"),
                "pr_auc": MulticlassAveragePrecision(num_classes=num_classes, average="macro"),
            }
        ),
        "f1_macro",
    )


class GraphDDIModule(LightningModule):
    """Two-tower GNN with an interaction map + MLP head."""

    def __init__(
        self,
        encoder: nn.Module,
        readout: nn.Module,
        hidden_dim: int,
        mlp_hidden_dim: int,
        task: Task = "multiclass",
        num_classes: int = 86,
        use_interaction_map: bool = True,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 1000,
        max_steps: int = 100_000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "readout"])

        self.encoder = encoder
        self.readout = readout
        self.use_interaction_map = use_interaction_map
        self.task = task

        head_in = 2 * int(readout.out_channels)  # type: ignore[arg-type]
        head_out = 1 if task == "binary" else num_classes
        self.head = nn.Sequential(
            nn.Linear(head_in, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, head_out),
        )

        # Pattern A: logits go into the metric collection as-is — torchmetrics 1.x
        # applies argmax (F1/P/R) or softmax (AUROC/AP) internally.
        metrics, self._best_key = _build_metrics(task, num_classes)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_loss_best = MinMetric()
        self.val_metric_best = MaxMetric()

        if task == "multiclass":
            self.test_per_class_f1 = MulticlassF1Score(num_classes=num_classes, average=None)

    # ------- core forward --------------------------------------------------
    def _encode_pair(
        self, batch_a, batch_b
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_a = self.encoder(batch_a.x, batch_a.edge_index, batch_a.edge_attr, batch_a.batch)
        h_b = self.encoder(batch_b.x, batch_b.edge_index, batch_b.edge_attr, batch_b.batch)
        if self.use_interaction_map:
            a_prime, b_prime = interaction_map(h_a, batch_a.batch, h_b, batch_b.batch)
        else:
            # Fair-shape ablation: cat A with itself so readout input stays 2*hidden.
            a_prime, b_prime = h_a, h_b
        h_a = torch.cat([h_a, a_prime], dim=-1)
        h_b = torch.cat([h_b, b_prime], dim=-1)
        return h_a, batch_a.batch, h_b, batch_b.batch

    def forward(self, batch_a, batch_b) -> torch.Tensor:
        h_a, ba, h_b, bb = self._encode_pair(batch_a, batch_b)
        g_a = self.readout(h_a, ba)
        g_b = self.readout(h_b, bb)
        return self.head(torch.cat([g_a, g_b], dim=-1))

    # ------- loss + steps --------------------------------------------------
    def _step(self, batch, stage: str):
        batch_a, batch_b, y = batch
        bs = y.size(0)
        logits = self(batch_a, batch_b)

        if self.task == "binary":
            logits = logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            target = y.int()
        else:
            loss = F.cross_entropy(
                logits, y.long(), label_smoothing=self.hparams.label_smoothing
            )
            target = y.long()

        loss_metric = getattr(self, f"{stage}_loss")
        loss_metric.update(loss.detach())
        self.log(
            f"{stage}/loss",
            loss_metric,
            on_epoch=True,
            on_step=(stage == "train"),
            prog_bar=True,
            sync_dist=True,
            batch_size=bs,
        )

        metrics = getattr(self, f"{stage}_metrics")
        metrics.update(logits, target)
        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True, batch_size=bs)

        if stage == "test" and self.task == "multiclass":
            self.test_per_class_f1.update(logits, y.long())

        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self.val_loss_best.update(self.val_loss.compute())
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True)

        self.val_metric_best.update(self.val_metrics[self._best_key].compute())
        suffix = "macro_f1_best" if self.task == "multiclass" else "f1_best"
        self.log(f"val/{suffix}", self.val_metric_best.compute(), sync_dist=True)

    def test_step(self, batch, _):
        self._step(batch, "test")

    def on_test_epoch_end(self) -> None:
        if self.task != "multiclass":
            return
        for c, f1 in enumerate(self.test_per_class_f1.compute().tolist()):
            self.log(f"test/f1_class_{c}", f1, sync_dist=True)
        self.test_per_class_f1.reset()

    def configure_optimizers(self):
        # Do NOT enable AdamW(fused=True): Lightning's AMP plugin refuses to clip
        # gradients when the optimizer unscales internally, so fused + bf16-mixed
        # + gradient_clip_val crashes at step 1.
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup = max(1, self.hparams.warmup_steps)
        total = max(self.hparams.max_steps, warmup + 1)
        sched = SequentialLR(
            opt,
            schedulers=[
                LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(opt, T_max=total - warmup),
            ],
            milestones=[warmup],
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
