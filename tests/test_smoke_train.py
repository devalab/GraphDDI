"""Smoke training runs on the mini fixture (CPU-only and CUDA-only)."""

import pytest
import torch
from lightning import Trainer

from tests.conftest import make_tiny_model


def _cpu_trainer(**kwargs) -> Trainer:
    """Trainer with all the bells silenced, sized for unit tests."""
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


def test_train_few_steps_is_finite_with_full_metric_suite(multiclass_dm):
    model = make_tiny_model(multiclass_dm)
    trainer = _cpu_trainer(max_steps=5, log_every_n_steps=1)
    trainer.fit(model, datamodule=multiclass_dm)

    cb = trainer.callback_metrics
    final_loss = cb.get("train/loss_epoch") or cb.get("train/loss")
    assert final_loss is not None and torch.isfinite(torch.tensor(float(final_loss)))

    # The metric collection IS the source of truth for which keys appear.
    expected = set(model.train_metrics)
    assert not (expected - set(cb)), f"missing metric keys: {expected - set(cb)}"


def test_val_loss_best_is_populated_after_validation_epoch(multiclass_dm):
    """One full epoch populates val/loss_best and val/macro_f1_best."""
    model = make_tiny_model(multiclass_dm)
    trainer = _cpu_trainer(max_epochs=1, limit_train_batches=2, limit_val_batches=2)
    trainer.fit(model, datamodule=multiclass_dm)
    cb = trainer.callback_metrics
    assert {"val/loss_best", "val/macro_f1_best"} <= set(cb)
    assert torch.isfinite(cb["val/loss_best"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16-mixed requires CUDA")
def test_bf16_mixed_with_gradient_clipping_does_not_crash(multiclass_dm):
    """Regression: AMP + ``gradient_clip_val`` + ``AdamW(fused=True)`` crashed at step 1
    ("optimizer does not allow for gradient clipping"). Production configs use this
    combo, so this exercises the path the CPU smoke skips."""
    model = make_tiny_model(multiclass_dm)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=multiclass_dm)
    assert "train/loss_epoch" in trainer.callback_metrics
