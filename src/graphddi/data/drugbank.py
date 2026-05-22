"""DrugBank DDI DataModule (binary + multiclass)."""

import logging
import pickle
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from graphddi.data._loader import quiet_stderr
from graphddi.data.collate import PairListDataset, maybe_compute_in_degree_histogram, pair_collate
from graphddi.data.featurizer import smiles_to_graph

log = logging.getLogger(__name__)

Task = Literal["binary", "multiclass"]


class DrugBankDataModule(LightningDataModule):
    """DrugBank DDI via TDC.

    The TDC ``DDI(name="DrugBank")`` loader downloads to ``data_root``. SMILES
    that RDKit cannot parse are dropped. For the multiclass task, labels 1..86
    are converted to 0..85 and classes below ``oversample_min_fraction`` of the
    training set are oversampled. Validation and test splits are never
    oversampled.

    For the binary task, negative samples are drawn at random from
    non-interacting drug pairs, sized to match the positive set.
    """

    num_classes: int
    class_label_map: dict[int, str]
    train_in_degree_histogram: torch.Tensor

    def __init__(
        self,
        task: Task = "multiclass",
        batch_size: int = 256,
        num_workers: int = 4,
        data_root: str = "./data",
        seed: int = 42,
        split_fracs: Sequence[float] = (0.7, 0.1, 0.2),
        oversample_min_fraction: float = 0.005,
        cache: bool = True,
    ) -> None:
        super().__init__()
        if task not in ("binary", "multiclass"):
            raise ValueError(f"unknown task {task!r}")
        self.save_hyperparameters()
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = Path(data_root)
        self.seed = seed
        self.split_fracs = list(split_fracs)
        self.oversample_min_fraction = oversample_min_fraction
        self.cache = cache

        # filled in by setup()
        self._train: PairListDataset | None = None
        self._val: PairListDataset | None = None
        self._test: PairListDataset | None = None
        self.num_classes = 86 if task == "multiclass" else 1
        self.class_label_map = {}

    # ----- TDC download (called once per node) ---------------------------------
    def prepare_data(self) -> None:
        from tdc.multi_pred import DDI

        self.data_root.mkdir(parents=True, exist_ok=True)
        with quiet_stderr():
            DDI(name="DrugBank", path=str(self.data_root))

    # ----- Setup ----------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return

        cache_dir = self.data_root / "drugbank" / "featurized"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"splits_{self.task}_seed{self.seed}.pkl"

        if self.cache and cache_path.exists():
            log.info("Loading cached DrugBank splits from %s", cache_path)
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            self._train = payload["train"]
            self._val = payload["val"]
            self._test = payload["test"]
            self.num_classes = payload["num_classes"]
            self.class_label_map = payload["class_label_map"]
            self.train_in_degree_histogram = payload["deg"]
            return

        from tdc.multi_pred import DDI
        from tdc.utils import get_label_map

        with quiet_stderr():
            ddi = DDI(name="DrugBank", path=str(self.data_root))
            split = ddi.get_split(method="random", seed=self.seed, frac=list(self.split_fracs))

        train_df: pd.DataFrame = split["train"]
        val_df: pd.DataFrame = split["valid"]
        test_df: pd.DataFrame = split["test"]

        # Featurize every unique SMILES once.
        smiles_to_id: dict[str, str] = {}
        for df in (train_df, val_df, test_df):
            for did, smi in zip(df["Drug1_ID"], df["Drug1"], strict=True):
                smiles_to_id[smi] = did
            for did, smi in zip(df["Drug2_ID"], df["Drug2"], strict=True):
                smiles_to_id[smi] = did

        graph_by_id: dict[str, Data] = {}
        bad_ids: set[str] = set()
        for smi, did in smiles_to_id.items():
            g = smiles_to_graph(smi)
            if g is None:
                bad_ids.add(did)
            else:
                graph_by_id[did] = g
        if bad_ids:
            log.warning("Dropping %d drugs whose SMILES RDKit cannot parse", len(bad_ids))

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            return df[~df["Drug1_ID"].isin(bad_ids) & ~df["Drug2_ID"].isin(bad_ids)].reset_index(
                drop=True
            )

        train_df = _filter(train_df)
        val_df = _filter(val_df)
        test_df = _filter(test_df)

        self.class_label_map = get_label_map(name="DrugBank", task="DDI")

        if self.task == "multiclass":
            self.num_classes = 86
            train_pairs = self._oversample_multiclass(train_df, graph_by_id)
            val_pairs = self._build_multiclass_pairs(val_df, graph_by_id)
            test_pairs = self._build_multiclass_pairs(test_df, graph_by_id)
        else:
            self.num_classes = 1
            all_known_pairs: set[tuple[str, str]] = set()
            for df in (train_df, val_df, test_df):
                for d1, d2 in zip(df["Drug1_ID"], df["Drug2_ID"], strict=True):
                    all_known_pairs.add((d1, d2))
                    all_known_pairs.add((d2, d1))
            all_drugs = sorted(graph_by_id.keys())
            train_pairs = self._build_binary_pairs(
                train_df, graph_by_id, all_drugs, all_known_pairs, self.seed
            )
            val_pairs = self._build_binary_pairs(
                val_df, graph_by_id, all_drugs, all_known_pairs, self.seed + 1
            )
            test_pairs = self._build_binary_pairs(
                test_df, graph_by_id, all_drugs, all_known_pairs, self.seed + 2
            )

        self._train = PairListDataset(*train_pairs)
        self._val = PairListDataset(*val_pairs)
        self._test = PairListDataset(*test_pairs)

        self.train_in_degree_histogram = maybe_compute_in_degree_histogram(
            list(graph_by_id.values())
        )

        if self.cache:
            with cache_path.open("wb") as f:
                pickle.dump(
                    {
                        "train": self._train,
                        "val": self._val,
                        "test": self._test,
                        "num_classes": self.num_classes,
                        "class_label_map": self.class_label_map,
                        "deg": self.train_in_degree_histogram,
                    },
                    f,
                )

    # ----- Pair builders -------------------------------------------------------
    @staticmethod
    def _build_multiclass_pairs(
        df: pd.DataFrame, graph_by_id: dict[str, Data]
    ) -> tuple[list[Data], list[Data], list[int]]:
        a, b, y = [], [], []
        for d1, d2, label in zip(df["Drug1_ID"], df["Drug2_ID"], df["Y"], strict=True):
            if d1 not in graph_by_id or d2 not in graph_by_id:
                continue
            a.append(graph_by_id[d1])
            b.append(graph_by_id[d2])
            y.append(int(label) - 1)  # TDC labels are 1..86 → 0..85
        return a, b, y

    def _oversample_multiclass(
        self, train_df: pd.DataFrame, graph_by_id: dict[str, Data]
    ) -> tuple[list[Data], list[Data], list[int]]:
        labels = (train_df["Y"].astype(int) - 1).to_numpy()
        d1 = train_df["Drug1_ID"].to_numpy()
        d2 = train_df["Drug2_ID"].to_numpy()

        # Build the initial set of valid (graph-parsable) indices.
        valid_idx = [i for i in range(len(labels)) if d1[i] in graph_by_id and d2[i] in graph_by_id]
        by_class: dict[int, list[int]] = defaultdict(list)
        for i in valid_idx:
            by_class[int(labels[i])].append(i)

        rng = np.random.default_rng(self.seed)
        oversampled_idx: list[int] = list(valid_idx)
        # One-shot oversampling against the *initial* training size. Iterating
        # to a post-oversample floor can be unsatisfiable when ``k * floor >= 1``
        # (a runaway growth bug). Treat "at least 0.5% of the training data" as
        # "at least 0.5% of the original training data" — the natural reading.
        target = max(1, int(np.ceil(len(valid_idx) * self.oversample_min_fraction)))
        for idxs in by_class.values():
            need = target - len(idxs)
            if need <= 0:
                continue
            extra = rng.choice(idxs, size=need, replace=True)
            oversampled_idx.extend(int(x) for x in extra)

        rng.shuffle(oversampled_idx)
        a, b, y = [], [], []
        for i in oversampled_idx:
            a.append(graph_by_id[d1[i]])
            b.append(graph_by_id[d2[i]])
            y.append(int(labels[i]))
        return a, b, y

    @staticmethod
    def _build_binary_pairs(
        df: pd.DataFrame,
        graph_by_id: dict[str, Data],
        all_drugs: list[str],
        known_positive: set[tuple[str, str]],
        seed: int,
    ) -> tuple[list[Data], list[Data], list[int]]:
        rng = np.random.default_rng(seed)
        a: list[Data] = []
        b: list[Data] = []
        y: list[int] = []

        positive_pairs: list[tuple[str, str]] = []
        for d1, d2 in zip(df["Drug1_ID"], df["Drug2_ID"], strict=True):
            if d1 in graph_by_id and d2 in graph_by_id:
                positive_pairs.append((d1, d2))

        for d1, d2 in positive_pairs:
            a.append(graph_by_id[d1])
            b.append(graph_by_id[d2])
            y.append(1)

        # Negative sampling: pick random pairs not in the known-positive set.
        seen_neg: set[tuple[str, str]] = set()
        n_target = len(positive_pairs)
        attempts = 0
        max_attempts = n_target * 50
        while len(seen_neg) < n_target and attempts < max_attempts:
            attempts += 1
            i, j = rng.integers(0, len(all_drugs), size=2)
            if i == j:
                continue
            d1, d2 = all_drugs[i], all_drugs[j]
            key = (d1, d2)
            if key in known_positive or key in seen_neg:
                continue
            seen_neg.add(key)
            a.append(graph_by_id[d1])
            b.append(graph_by_id[d2])
            y.append(0)
        return a, b, y

    # ----- Dataloaders ---------------------------------------------------------
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

    # ----- Helpers exposed for analysis ---------------------------------------
    def train_class_counts(self) -> Counter[int]:
        assert self._train is not None
        return Counter(self._train.labels)
