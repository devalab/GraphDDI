"""Forward-pass sanity checks for GraphDDI."""

import pytest
import torch

from graphddi.models import (
    GraphDDIModule,
    MPNNEncoder,
    PNAEncoder,
    Set2SetReadout,
    SetTransformerReadout,
)
from tests.conftest import EDGE_DIM, HIDDEN, NODE_DIM


def _build(dm, *, task, use_imap, encoder_cls, readout_cls):
    """Build a tiny model with the requested encoder/readout/task combo."""
    encoder_kwargs = dict(
        in_channels=NODE_DIM,
        hidden_channels=HIDDEN,
        edge_dim=EDGE_DIM,
        edge_hidden_channels=2 * HIDDEN,
        num_layers=2,
    )
    if encoder_cls is PNAEncoder:
        encoder_kwargs["deg"] = dm.train_in_degree_histogram

    readout_in = 2 * HIDDEN  # after [A | A'] cat, even in no-IMAP fair-shape mode
    readout_kwargs = (
        dict(in_channels=readout_in, processing_steps=2, project_to=readout_in)
        if readout_cls is Set2SetReadout
        else dict(channels=readout_in, project_to=readout_in)
    )

    return GraphDDIModule(
        encoder=encoder_cls(**encoder_kwargs),
        readout=readout_cls(**readout_kwargs),
        hidden_dim=HIDDEN,
        mlp_hidden_dim=32,
        task=task,
        num_classes=86 if task == "multiclass" else 1,
        use_interaction_map=use_imap,
    )


@pytest.mark.parametrize(
    "encoder_cls,readout_cls,use_imap",
    [
        (PNAEncoder, Set2SetReadout, True),
        (PNAEncoder, Set2SetReadout, False),
        (MPNNEncoder, Set2SetReadout, True),
        (PNAEncoder, SetTransformerReadout, True),
    ],
    ids=["pna+imap+s2s", "pna+noimap+s2s", "mpnn+imap+s2s", "pna+imap+sett"],
)
def test_forward_output_shape_multiclass(mini_batch, encoder_cls, readout_cls, use_imap):
    dm, batch_a, batch_b, _ = mini_batch
    model = _build(
        dm, task="multiclass", use_imap=use_imap, encoder_cls=encoder_cls, readout_cls=readout_cls
    ).eval()
    with torch.no_grad():
        out = model(batch_a, batch_b)
    assert out.shape == (batch_a.num_graphs, 86)
    assert torch.isfinite(out).all()


def test_binary_output_shape(binary_mini_batch):
    dm, batch_a, batch_b, _ = binary_mini_batch
    model = _build(
        dm, task="binary", use_imap=True, encoder_cls=PNAEncoder, readout_cls=Set2SetReadout
    ).eval()
    with torch.no_grad():
        out = model(batch_a, batch_b)
    assert out.shape == (batch_a.num_graphs, 1)


def test_no_imap_preserves_fair_shape(mini_batch):
    """No-IMAP path concatenates [A | A] so the readout input stays at 2*hidden."""
    dm, batch_a, batch_b, _ = mini_batch
    model = _build(
        dm, task="multiclass", use_imap=False, encoder_cls=PNAEncoder, readout_cls=Set2SetReadout
    ).eval()
    with torch.no_grad():
        h_a, _, h_b, _ = model._encode_pair(batch_a, batch_b)
    assert h_a.shape[-1] == h_b.shape[-1] == 2 * HIDDEN


def test_pna_encoder_has_per_layer_graphnorm_and_gru(mini_batch):
    dm, *_ = mini_batch
    enc = PNAEncoder(
        in_channels=NODE_DIM,
        hidden_channels=HIDDEN,
        edge_dim=EDGE_DIM,
        edge_hidden_channels=2 * HIDDEN,
        num_layers=4,
        deg=dm.train_in_degree_histogram,
    )
    assert len(enc.convs) == len(enc.norms) == len(enc.grus) == 4
    assert all(isinstance(g, torch.nn.GRUCell) for g in enc.grus)


def test_full_multiclass_model_param_count_within_5pct_of_906k():
    """Production-dim model lands near 906 K params (target from the 2024 log)."""
    deg = torch.tensor([200, 10511, 20325, 14377, 1248, 2, 1], dtype=torch.long)
    encoder = PNAEncoder(
        in_channels=108,
        hidden_channels=64,
        edge_dim=9,
        edge_hidden_channels=128,
        num_layers=6,
        deg=deg,
        dropout=0.1,
    )
    readout = Set2SetReadout(in_channels=128, processing_steps=2, project_to=128)
    model = GraphDDIModule(
        encoder=encoder,
        readout=readout,
        hidden_dim=64,
        mlp_hidden_dim=128,
        task="multiclass",
        num_classes=86,
        use_interaction_map=True,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target, tolerance = 906_000, 0.05
    assert (1 - tolerance) * target <= n_params <= (1 + tolerance) * target, (
        f"param count {n_params} outside ±5% of {target}"
    )
