"""DataModule backed by a tiny CSV fixture for fast tests / smoke runs."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from graphddi.data.collate import PairListDataset, maybe_compute_in_degree_histogram, pair_collate
from graphddi.data.featurizer import smiles_to_graph

Task = Literal["binary", "multiclass"]


class DrugBankMiniDataModule(LightningDataModule):
    """Reads ``tests/fixtures/drugbank_mini.csv`` (a real stratified slice of TDC
    DrugBank) and exposes the same interface as :class:`DrugBankDataModule`.

    Columns: ``Drug1_ID, Drug1, Drug2_ID, Drug2, Y`` (Y in 1..86, original
    TDC convention).
    """

    num_classes: int
    train_in_degree_histogram: torch.Tensor
    class_label_map: dict[int, str]

    def __init__(
        self,
        csv_path: str = "tests/fixtures/drugbank_mini.csv",
        task: Task = "multiclass",
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 42,
        split_fracs: Sequence[float] = (0.7, 0.1, 0.2),
        oversample_min_fraction: float = 0.0,  # disabled for the mini fixture
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.csv_path = Path(csv_path)
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.split_fracs = list(split_fracs)
        self.oversample_min_fraction = oversample_min_fraction

        self._train: PairListDataset | None = None
        self._val: PairListDataset | None = None
        self._test: PairListDataset | None = None
        self.num_classes = 86 if task == "multiclass" else 1
        self.class_label_map = {}

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Mini fixture not found at {self.csv_path}. "
                "Run `python scripts/prepare_test_fixture.py` first."
            )

        df = pd.read_csv(self.csv_path)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(df))
        df = df.iloc[perm].reset_index(drop=True)

        n = len(df)
        n_train = int(self.split_fracs[0] * n)
        n_val = int(self.split_fracs[1] * n)
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train : n_train + n_val]
        test_df = df.iloc[n_train + n_val :]

        graphs = {}
        for _, row in df.iterrows():
            for did, smi in [(row["Drug1_ID"], row["Drug1"]), (row["Drug2_ID"], row["Drug2"])]:
                if did in graphs:
                    continue
                g = smiles_to_graph(smi)
                if g is not None:
                    graphs[did] = g

        if self.task == "multiclass":
            self._train = self._build(train_df, graphs, multiclass=True, oversample=True)
            self._val = self._build(val_df, graphs, multiclass=True, oversample=False)
            self._test = self._build(test_df, graphs, multiclass=True, oversample=False)
        else:
            all_drugs = sorted(graphs.keys())
            known: set[tuple[str, str]] = set()
            for _, r in df.iterrows():
                known.add((r["Drug1_ID"], r["Drug2_ID"]))
                known.add((r["Drug2_ID"], r["Drug1_ID"]))
            self._train = self._build_binary(train_df, graphs, all_drugs, known, self.seed)
            self._val = self._build_binary(val_df, graphs, all_drugs, known, self.seed + 1)
            self._test = self._build_binary(test_df, graphs, all_drugs, known, self.seed + 2)

        self.train_in_degree_histogram = maybe_compute_in_degree_histogram(list(graphs.values()))

    def _build(self, df, graphs, multiclass: bool, oversample: bool = False) -> PairListDataset:
        a, b, y = [], [], []
        for _, r in df.iterrows():
            if r["Drug1_ID"] not in graphs or r["Drug2_ID"] not in graphs:
                continue
            a.append(graphs[r["Drug1_ID"]])
            b.append(graphs[r["Drug2_ID"]])
            if multiclass:
                y.append(int(r["Y"]) - 1)
            else:
                y.append(1)
        if not multiclass or not oversample or self.oversample_min_fraction <= 0 or not y:
            return PairListDataset(a, b, y)

        # One-shot oversampling: ensure each class has at least
        # ``ceil(N_initial * fraction)`` samples in train. Iterating to a
        # post-oversample fixed point can be unsatisfiable when many classes
        # share a high floor (k*floor >= 1), so we use the simpler invariant
        # against the *initial* training size — which matches the natural
        # reading of "at least 0.5% of the training data".
        by_class: dict[int, list[int]] = defaultdict(list)
        for idx, lab in enumerate(y):
            by_class[lab].append(idx)
        target = max(1, int(np.ceil(len(y) * self.oversample_min_fraction)))
        rng = np.random.default_rng(self.seed)
        for cls, idxs in list(by_class.items()):
            need = target - len(idxs)
            if need <= 0:
                continue
            for ei in rng.choice(idxs, size=need, replace=True):
                a.append(a[ei])
                b.append(b[ei])
                y.append(cls)
        return PairListDataset(a, b, y)

    @staticmethod
    def _build_binary(df, graphs, all_drugs, known, seed) -> PairListDataset:
        rng = np.random.default_rng(seed)
        a, b, y = [], [], []
        for _, r in df.iterrows():
            if r["Drug1_ID"] not in graphs or r["Drug2_ID"] not in graphs:
                continue
            a.append(graphs[r["Drug1_ID"]])
            b.append(graphs[r["Drug2_ID"]])
            y.append(1)
        n_target = sum(1 for v in y if v == 1)
        seen_neg: set[tuple[str, str]] = set()
        attempts = 0
        while len(seen_neg) < n_target and attempts < n_target * 100:
            attempts += 1
            i, j = rng.integers(0, len(all_drugs), size=2)
            if i == j:
                continue
            d1, d2 = all_drugs[i], all_drugs[j]
            if (d1, d2) in known or (d1, d2) in seen_neg:
                continue
            seen_neg.add((d1, d2))
            a.append(graphs[d1])
            b.append(graphs[d2])
            y.append(0)
        return PairListDataset(a, b, y)

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=pair_collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False)

    def train_class_counts(self) -> Counter[int]:
        assert self._train is not None
        return Counter(self._train.labels)
