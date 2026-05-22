"""Set2SetReadout projection (post-readout linear) sanity checks."""

import torch

from graphddi.models import Set2SetReadout, SetTransformerReadout


def test_set2set_no_projection_doubles_in_channels():
    rd = Set2SetReadout(in_channels=8, processing_steps=2)
    x = torch.randn(7, 8)                      # 7 nodes
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    out = rd(x, batch)
    assert rd.out_channels == 16               # Set2Set produces 2 * in_channels
    assert out.shape == (3, 16)                # 3 graphs in this batch


def test_set2set_with_projection_reports_project_to():
    rd = Set2SetReadout(in_channels=8, processing_steps=2, project_to=4)
    x = torch.randn(7, 8)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    out = rd(x, batch)
    assert rd.out_channels == 4
    assert out.shape == (3, 4)


def test_set2set_projection_is_relu_activated():
    """Projection module exposes the Linear + ReLU pipeline used in the spec."""
    rd = Set2SetReadout(in_channels=8, processing_steps=2, project_to=4)
    assert isinstance(rd.proj, torch.nn.Sequential), "projection should be a Sequential"
    assert isinstance(rd.proj[0], torch.nn.Linear), "first step should be Linear"
    assert isinstance(rd.proj[1], torch.nn.ReLU), "second step should be ReLU"


def test_settransformer_with_projection_reports_project_to():
    rd = SetTransformerReadout(channels=8, project_to=4)
    x = torch.randn(6, 8)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    out = rd(x, batch)
    assert rd.out_channels == 4
    assert out.shape == (2, 4)


def test_settransformer_no_projection_reports_channels():
    rd = SetTransformerReadout(channels=8)
    assert rd.out_channels == 8
    x = torch.randn(6, 8)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    out = rd(x, batch)
    assert out.shape == (2, 8)
