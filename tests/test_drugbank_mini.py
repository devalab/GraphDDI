"""Smoke tests for the mini DataModule using a real (tiny) TDC slice."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from graphddi.data.featurizer import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from graphddi.data.mini import DrugBankMiniDataModule

FIXTURE = Path(__file__).parent / "fixtures" / "drugbank_mini.csv"


@pytest.fixture
def multiclass_dm() -> DrugBankMiniDataModule:
    dm = DrugBankMiniDataModule(csv_path=str(FIXTURE), task="multiclass", batch_size=4)
    dm.setup()
    return dm


@pytest.fixture
def binary_dm() -> DrugBankMiniDataModule:
    dm = DrugBankMiniDataModule(csv_path=str(FIXTURE), task="binary", batch_size=4)
    dm.setup()
    return dm


def test_fixture_present():
    assert FIXTURE.exists(), (
        f"Fixture missing at {FIXTURE}. Run `python scripts/prepare_test_fixture.py` once."
    )


def test_multiclass_batch_shapes(multiclass_dm):
    loader = multiclass_dm.train_dataloader()
    batch_a, batch_b, labels = next(iter(loader))
    assert batch_a.x.shape[1] == NODE_FEATURE_DIM
    assert batch_b.x.shape[1] == NODE_FEATURE_DIM
    if batch_a.edge_attr.numel() > 0:
        assert batch_a.edge_attr.shape[1] == EDGE_FEATURE_DIM
    if batch_b.edge_attr.numel() > 0:
        assert batch_b.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert labels.dtype in (torch.int64, torch.long, torch.int32)
    assert labels.shape == (batch_a.batch.max().item() + 1,)
    # Labels for multiclass are 0..85.
    assert (labels >= 0).all() and (labels < 86).all()


def test_binary_batch_labels(binary_dm):
    loader = binary_dm.train_dataloader()
    batch_a, batch_b, labels = next(iter(loader))
    # Binary labels are 0 or 1.
    unique = set(labels.tolist())
    assert unique <= {0, 1}


def test_paired_batches_have_independent_indices(multiclass_dm):
    """``batch_a.batch`` indexes nodes within batch A only — must not collide
    with batch B's node space."""
    loader = multiclass_dm.train_dataloader()
    batch_a, batch_b, _ = next(iter(loader))
    # Both batches should have the same number of graphs (same batch size).
    assert batch_a.batch.max().item() == batch_b.batch.max().item()
    # Node indices live in independent 0..N_a-1 and 0..N_b-1 ranges, not summed.
    assert batch_a.x.shape[0] >= batch_a.batch.max().item() + 1
    assert batch_b.x.shape[0] >= batch_b.batch.max().item() + 1


def test_splits_disjoint_by_size(multiclass_dm):
    n_train = len(multiclass_dm._train.labels)
    n_val = len(multiclass_dm._val.labels)
    n_test = len(multiclass_dm._test.labels)
    assert n_train > 0
    assert n_val >= 0
    assert n_test > 0
    assert n_train > n_val
    assert n_train > n_test


def test_in_degree_histogram_present(multiclass_dm):
    hist = multiclass_dm.train_in_degree_histogram
    assert hist.dim() == 1
    assert hist.sum() > 0
