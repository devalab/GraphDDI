"""Interaction map: A' = tanh(A B^T / sqrt(H)) @ B, B' = ...^T @ A (Eqs. 6–8)."""

import math

import torch
from torch_geometric.utils import to_dense_batch


def interaction_map(
    a: torch.Tensor,
    batch_a: torch.Tensor,
    b: torch.Tensor,
    batch_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_dense, mask_a = to_dense_batch(a, batch_a)  # [B, Na_max, H]
    b_dense, mask_b = to_dense_batch(b, batch_b)  # [B, Nb_max, H]

    a_dense = a_dense * mask_a.unsqueeze(-1)
    b_dense = b_dense * mask_b.unsqueeze(-1)

    hidden = a_dense.size(-1)
    inter = torch.tanh(a_dense @ b_dense.transpose(-1, -2) / math.sqrt(hidden))
    # Zero out interactions for padded positions in either A or B.
    inter = inter * mask_a.unsqueeze(-1) * mask_b.unsqueeze(-2)

    a_prime_dense = inter @ b_dense  # [B, Na_max, H]
    b_prime_dense = inter.transpose(-1, -2) @ a_dense  # [B, Nb_max, H]

    return a_prime_dense[mask_a], b_prime_dense[mask_b]
