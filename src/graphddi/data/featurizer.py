"""SMILES → PyG graph featurizer.

Node features = 108 dims, edge features = 9 dims, hydrogens dropped."""

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import Atom, Bond, BondStereo, BondType, ChiralType, HybridizationType
from torch_geometric.data import Data

# Mute RDKit's C++ parse-error chatter; we already return None on failure.
RDLogger.DisableLog("rdApp.*")

# Canonical periodic-table atom list used for the 64-dim one-hot. The order is
# fixed so that loading a saved checkpoint keeps the same feature indices.
ATOM_TYPES: list[str] = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
]  # 63 named + 1 "other" bucket = 64
assert len(ATOM_TYPES) == 63

CHIRAL_TAGS: list[ChiralType] = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    ChiralType.CHI_OTHER,
    ChiralType.CHI_TETRAHEDRAL,
    ChiralType.CHI_SQUAREPLANAR,
    ChiralType.CHI_OCTAHEDRAL,
]  # 7

# Degree buckets 0..9 plus an "other" bucket → 11.
DEGREES: list[int] = list(range(10))

# Formal charge buckets −5..+6 → 12.
FORMAL_CHARGES: list[int] = list(range(-5, 7))

# Hydrogen count buckets 0..4 → 5.
NUM_HS: list[int] = list(range(5))

HYBRIDIZATIONS: list[HybridizationType] = [
    HybridizationType.UNSPECIFIED,
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]  # 7

BOND_TYPES: list[BondType] = [
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
]  # 4

BOND_STEREOS: list[BondStereo] = [
    BondStereo.STEREONONE,
    BondStereo.STEREOE,
    BondStereo.STEREOZ,
    BondStereo.STEREOANY,
]  # 4

NODE_FEATURE_DIM: int = (
    (len(ATOM_TYPES) + 1)  # atom type one-hot incl. "other"
    + len(CHIRAL_TAGS)  # chiral
    + (len(DEGREES) + 1)  # degree incl. "other"
    + len(FORMAL_CHARGES)  # formal charge
    + len(NUM_HS)  # number of Hs
    + len(HYBRIDIZATIONS)  # hybridization
    + 1  # aromatic
    + 1  # in-ring
)
assert NODE_FEATURE_DIM == 108, f"expected 108-dim node features, got {NODE_FEATURE_DIM}"

EDGE_FEATURE_DIM: int = len(BOND_TYPES) + len(BOND_STEREOS) + 1
assert EDGE_FEATURE_DIM == 9, f"expected 9-dim edge features, got {EDGE_FEATURE_DIM}"


def _one_hot(value: object, choices: list, with_other: bool = False) -> list[float]:
    out = [0.0] * (len(choices) + (1 if with_other else 0))
    if value in choices:
        out[choices.index(value)] = 1.0
    elif with_other:
        out[-1] = 1.0
    return out


def _atom_features(atom: Atom) -> list[float]:
    return [
        *_one_hot(atom.GetSymbol(), ATOM_TYPES, with_other=True),
        *_one_hot(atom.GetChiralTag(), CHIRAL_TAGS),
        *_one_hot(atom.GetTotalDegree(), DEGREES, with_other=True),
        *_one_hot(atom.GetFormalCharge(), FORMAL_CHARGES),
        *_one_hot(min(atom.GetTotalNumHs(), 4), NUM_HS),
        *_one_hot(atom.GetHybridization(), HYBRIDIZATIONS),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ]


def _bond_features(bond: Bond) -> list[float]:
    return [
        *_one_hot(bond.GetBondType(), BOND_TYPES),
        *_one_hot(bond.GetStereo(), BOND_STEREOS),
        float(bond.GetIsConjugated()),
    ]


def smiles_to_graph(smiles: str) -> Data | None:
    """Convert a SMILES string to a PyG ``Data`` object. Returns ``None`` if RDKit
    cannot parse the input or the molecule has no atoms."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    edge_index_list: list[list[int]] = [[], []]
    edge_attr_list: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feats = _bond_features(bond)
        # Undirected: add both directions with the same edge features.
        edge_index_list[0].extend([i, j])
        edge_index_list[1].extend([j, i])
        edge_attr_list.extend([feats, feats])

    if len(edge_attr_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
