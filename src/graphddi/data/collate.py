"""Paired-drug batches: each sample is ``(data_a, data_b, label)`` and the
collate fn returns two independent PyG ``Batch`` objects plus the labels."""

import torch
from torch_geometric.data import Batch, Data

PairSample = tuple[Data, Data, int]
PairBatch = tuple[Batch, Batch, torch.Tensor]


def pair_collate(samples: list[PairSample]) -> PairBatch:
    batch_a = Batch.from_data_list([s[0] for s in samples])
    batch_b = Batch.from_data_list([s[1] for s in samples])
    labels = torch.tensor([s[2] for s in samples])
    return batch_a, batch_b, labels


class PairListDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        drug_a: list[Data],
        drug_b: list[Data],
        labels: list[int] | torch.Tensor,
    ) -> None:
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        assert len(drug_a) == len(drug_b) == len(labels)
        self.drug_a = drug_a
        self.drug_b = drug_b
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> PairSample:
        return self.drug_a[index], self.drug_b[index], self.labels[index]


def maybe_compute_in_degree_histogram(graphs: list[Data]) -> torch.Tensor:
    """In-degree histogram across all training graphs — PNAConv needs it."""
    degs = [
        torch.bincount(g.edge_index[1])
        for g in graphs
        if g.edge_index is not None and g.edge_index.numel() > 0
    ]
    if not degs:
        return torch.zeros(1, dtype=torch.long)
    max_deg = max(int(d.max()) for d in degs)
    hist = torch.zeros(max_deg + 1, dtype=torch.long)
    for d in degs:
        hist += torch.bincount(d, minlength=max_deg + 1)
    return hist
