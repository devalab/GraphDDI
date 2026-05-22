"""BioSNAP DDI DataModule (binary).

Splits follow CASTER (Huang et al., 2020); use ``scripts/download_biosnap.py``
to materialise ``data/biosnap/{train,val,test}.csv`` with columns
``SMILES1, SMILES2, Label``.
"""

import logging
from pathlib import Path

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from graphddi.data.collate import PairListDataset, maybe_compute_in_degree_histogram, pair_collate
from graphddi.data.featurizer import smiles_to_graph

log = logging.getLogger(__name__)


class BioSNAPDataModule(LightningDataModule):
    """Binary DDI on BioSNAP using the CASTER splits.

    Required files: ``<data_root>/biosnap/{train,val,test}.csv``. See module
    docstring for the download source.
    """

    num_classes: int = 1

    def __init__(
        self,
        data_root: str = "./data",
        batch_size: int = 256,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_root = Path(data_root) / "biosnap"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = "binary"
        self.num_classes = 1

        self._train: PairListDataset | None = None
        self._val: PairListDataset | None = None
        self._test: PairListDataset | None = None

    def prepare_data(self) -> None:
        for split in ("train", "val", "test"):
            if not (self.data_root / f"{split}.csv").exists():
                raise FileNotFoundError(
                    f"BioSNAP split not found: {self.data_root / f'{split}.csv'}. "
                    "See src/graphddi/data/biosnap.py docstring for download instructions."
                )

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return
        train_df = self._load_csv(self.data_root / "train.csv")
        val_df = self._load_csv(self.data_root / "val.csv")
        test_df = self._load_csv(self.data_root / "test.csv")

        # Featurize all unique SMILES once.
        graphs: dict[str, Data] = {}
        for df in (train_df, val_df, test_df):
            for smi in pd.concat([df["SMILES1"], df["SMILES2"]]).unique():
                if smi in graphs:
                    continue
                g = smiles_to_graph(smi)
                if g is not None:
                    graphs[smi] = g

        def build(df: pd.DataFrame) -> PairListDataset:
            a, b, y = [], [], []
            for _, r in df.iterrows():
                if r["SMILES1"] not in graphs or r["SMILES2"] not in graphs:
                    continue
                a.append(graphs[r["SMILES1"]])
                b.append(graphs[r["SMILES2"]])
                y.append(int(r["Label"]))
            return PairListDataset(a, b, y)

        self._train = build(train_df)
        self._val = build(val_df)
        self._test = build(test_df)
        self.train_in_degree_histogram = maybe_compute_in_degree_histogram(list(graphs.values()))

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        """Read a CASTER-style BioSNAP CSV, returning ``SMILES1, SMILES2, Label``.

        Accepts ``Drug1/Drug2`` or ``SMILES1/SMILES2`` for the columns and
        ``label/Label/Y`` for the target.
        """
        ALIASES = {
            "SMILES1": ("smiles1", "drug1"),
            "SMILES2": ("smiles2", "drug2"),
            "Label": ("label", "y"),
        }
        df = pd.read_csv(path)
        lower = {c.lower(): c for c in df.columns}
        rename = {
            lower[alias]: target
            for target, names in ALIASES.items()
            for alias in names
            if alias in lower and lower[alias] != target
        }
        df = df.rename(columns=rename)
        missing = set(ALIASES) - set(df.columns)
        if missing:
            raise ValueError(f"BioSNAP CSV {path} missing columns {missing}")
        return df

    def _loader(self, ds: PairListDataset | None, shuffle: bool) -> DataLoader:
        assert ds is not None, "setup() must be called before requesting a dataloader"
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=pair_collate,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False)
