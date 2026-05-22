"""Shared pytest fixtures: the mini DrugBank fixture path, datamodules, and a
tiny-model factory. Module-scoped so the CSV is parsed once per test file.
"""

from pathlib import Path

import pytest

from graphddi.data.mini import DrugBankMiniDataModule
from graphddi.models import GraphDDIModule, PNAEncoder, Set2SetReadout

FIXTURE = Path(__file__).parent / "fixtures" / "drugbank_mini.csv"

# Tiny dims for fast tests; production dims live in the configs.
NODE_DIM, EDGE_DIM, HIDDEN = 108, 9, 16


@pytest.fixture(scope="module")
def multiclass_dm() -> DrugBankMiniDataModule:
    dm = DrugBankMiniDataModule(csv_path=str(FIXTURE), task="multiclass", batch_size=4)
    dm.setup()
    return dm


@pytest.fixture(scope="module")
def binary_dm() -> DrugBankMiniDataModule:
    dm = DrugBankMiniDataModule(csv_path=str(FIXTURE), task="binary", batch_size=4)
    dm.setup()
    return dm


@pytest.fixture
def mini_batch(multiclass_dm):
    """One ``(dm, batch_a, batch_b, y)`` tuple from the multiclass loader."""
    batch_a, batch_b, y = next(iter(multiclass_dm.train_dataloader()))
    return multiclass_dm, batch_a, batch_b, y


@pytest.fixture
def binary_mini_batch(binary_dm):
    batch_a, batch_b, y = next(iter(binary_dm.train_dataloader()))
    return binary_dm, batch_a, batch_b, y


def make_tiny_model(dm: DrugBankMiniDataModule, task: str = "multiclass", **kwargs) -> GraphDDIModule:
    """Tiny PNA + Set2Set model used by the smoke tests."""
    encoder = PNAEncoder(
        in_channels=NODE_DIM,
        hidden_channels=HIDDEN,
        edge_dim=EDGE_DIM,
        edge_hidden_channels=2 * HIDDEN,
        num_layers=2,
        deg=dm.train_in_degree_histogram,
    )
    readout = Set2SetReadout(in_channels=2 * HIDDEN, processing_steps=2, project_to=2 * HIDDEN)
    return GraphDDIModule(
        encoder=encoder,
        readout=readout,
        hidden_dim=HIDDEN,
        mlp_hidden_dim=2 * HIDDEN,
        task=task,
        num_classes=86 if task == "multiclass" else 1,
        warmup_steps=2,
        max_steps=20,
        **kwargs,
    )
