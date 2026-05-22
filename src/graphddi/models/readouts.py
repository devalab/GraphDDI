"""Set2Set and SetTransformer readouts.

Both readouts optionally apply a ``Linear → ReLU`` post-projection so the
graph-level vector matches the head's expected input width without an extra
module outside the readout.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import Set2Set
from torch_geometric.nn.aggr import SetTransformerAggregation


class Set2SetReadout(nn.Module):
    """Set2Set with an optional ``Linear(2*in, project_to) → ReLU`` projection.

    Without ``project_to`` the output dim is ``2 * in_channels`` (Set2Set's
    native concat of context + last LSTM output).
    """

    def __init__(
        self,
        in_channels: int,
        processing_steps: int = 2,
        project_to: int | None = None,
    ) -> None:
        super().__init__()
        self.set2set = Set2Set(in_channels, processing_steps=processing_steps)
        set2set_dim = 2 * in_channels
        if project_to is not None:
            self.proj: nn.Module = nn.Sequential(
                nn.Linear(set2set_dim, project_to), nn.ReLU()
            )
            self.out_channels = project_to
        else:
            self.proj = nn.Identity()
            self.out_channels = set2set_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(self.set2set(x, index=batch))


class SetTransformerReadout(nn.Module):
    """SetTransformerAggregation with the same optional projection."""

    def __init__(
        self,
        channels: int,
        num_seed_points: int = 1,
        num_encoder_blocks: int = 2,
        num_decoder_blocks: int = 1,
        heads: int = 4,
        layer_norm: bool = True,
        dropout: float = 0.0,
        project_to: int | None = None,
    ) -> None:
        super().__init__()
        self.agg = SetTransformerAggregation(
            channels=channels,
            num_seed_points=num_seed_points,
            num_encoder_blocks=num_encoder_blocks,
            num_decoder_blocks=num_decoder_blocks,
            heads=heads,
            concat=False,
            layer_norm=layer_norm,
            dropout=dropout,
        )
        if project_to is not None:
            self.proj: nn.Module = nn.Sequential(
                nn.Linear(channels, project_to), nn.ReLU()
            )
            self.out_channels = project_to
        else:
            self.proj = nn.Identity()
            self.out_channels = channels

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() > 0 and (batch[1:] < batch[:-1]).any():
            order = batch.argsort()
            x = x[order]
            batch = batch[order]
        return self.proj(self.agg(x, index=batch))
