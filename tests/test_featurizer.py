"""Tests for the SMILES → PyG graph featurizer."""

import torch

from graphddi.data.featurizer import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, smiles_to_graph

# A mix of small drug-like molecules, aromatic systems, charged atoms, and rings.
KNOWN_SMILES = [
    "CC(=O)NC1=CC=C(O)C=C1",  # Acetaminophen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "C(C(=O)O)N",  # Glycine
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",  # Glucose (with stereo)
    "[Na+].[Cl-]",  # Disconnected ions (still parses)
    "C1CCCCC1",  # Cyclohexane
    "c1ccccc1",  # Benzene (lowercase aromatic)
    "N",  # Ammonia — single atom, no bonds
]


def test_dimensions_constants():
    assert NODE_FEATURE_DIM == 108
    assert EDGE_FEATURE_DIM == 9


def test_known_smiles_parse_to_valid_graphs():
    for smi in KNOWN_SMILES:
        data = smiles_to_graph(smi)
        assert data is not None, f"failed to parse {smi}"
        assert data.x.shape[1] == NODE_FEATURE_DIM, f"bad node dim for {smi}"
        assert data.x.shape[0] > 0
        # Ammonia has a single atom and no bonds, so edge_attr is empty.
        if data.edge_attr.shape[0] > 0:
            assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
        # No NaNs or negative entries (everything is one-hot or binary).
        assert not torch.isnan(data.x).any()
        assert (data.x >= 0).all()


def test_invalid_smiles_returns_none():
    assert smiles_to_graph("not a smiles!!!") is None
    assert smiles_to_graph("") is None


def test_edges_are_undirected():
    data = smiles_to_graph("CCO")  # ethanol — 2 bonds → 4 directed edges
    assert data is not None
    assert data.edge_index.shape[1] == 4
    # Every directed edge should have a matching reverse edge.
    edges = {tuple(e.tolist()) for e in data.edge_index.t()}
    for a, b in list(edges):
        assert (b, a) in edges
