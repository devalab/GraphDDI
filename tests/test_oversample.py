"""Tests that oversampling lifts minority classes to the target floor without
contaminating val/test splits."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from graphddi.data.mini import DrugBankMiniDataModule

FIXTURE = Path(__file__).parent / "fixtures" / "drugbank_mini.csv"


def test_oversample_lifts_minority_classes_to_floor():
    """Each class should hit `ceil(N_initial * floor)` after oversampling.

    The mini fixture has ~50 classes, so the per-class floor must satisfy
    `k * floor < 1` for the invariant to be feasible. We pick 0.5% — at 50
    classes that's 25%, well under 100%.
    """
    floor = 0.005  # 0.5% per-class floor
    dm_no = DrugBankMiniDataModule(
        csv_path=str(FIXTURE),
        task="multiclass",
        batch_size=4,
        oversample_min_fraction=0.0,
    )
    dm_no.setup()
    n_train_initial = len(dm_no._train.labels)

    dm = DrugBankMiniDataModule(
        csv_path=str(FIXTURE),
        task="multiclass",
        batch_size=4,
        oversample_min_fraction=floor,
    )
    dm.setup()
    counts: Counter[int] = dm.train_class_counts()
    target = math.ceil(n_train_initial * floor)
    for cls, cnt in counts.items():
        assert cnt >= target, f"class {cls} has only {cnt} (need ≥ {target})"


def test_val_test_are_not_oversampled():
    dm_no = DrugBankMiniDataModule(
        csv_path=str(FIXTURE), task="multiclass", batch_size=4, oversample_min_fraction=0.0
    )
    dm_no.setup()
    n_val_no = len(dm_no._val.labels)
    n_test_no = len(dm_no._test.labels)

    dm_yes = DrugBankMiniDataModule(
        csv_path=str(FIXTURE), task="multiclass", batch_size=4, oversample_min_fraction=0.01
    )
    dm_yes.setup()
    assert len(dm_yes._val.labels) == n_val_no
    assert len(dm_yes._test.labels) == n_test_no
    assert len(dm_yes._train.labels) >= len(dm_no._train.labels)
