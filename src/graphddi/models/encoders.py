"""PNA and MPNN encoders.

Both encoders share the same backbone — global node and edge embedding,
``num_layers`` of (``Conv`` → ``GraphNorm`` → ReLU → Dropout → ``GRUCell``),
and a final ``Linear`` + ``ReLU`` projection — so the loop body and the
scaffolding live on a small private base. Subclasses only inject the conv
operator via a factory callable.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm, NNConv, PNAConv

ConvFactory = Callable[[int, int], nn.Module]
"""``(hidden_channels, edge_hidden_channels) -> conv`` — builds one layer."""


class _GRUMessagePassingStack(nn.Module):
    """Shared backbone: emb → N × (conv → norm → act → drop → gru) → fc/act.

    Subclasses pass a ``conv_factory`` callable that builds one conv layer
    given ``(hidden_channels, edge_hidden_channels)``.
    """

    def __init__(
        self,
        conv_factory: ConvFactory,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        edge_hidden_channels: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_emb = nn.Linear(in_channels, hidden_channels)
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_channels), nn.ReLU()
        )
        self.convs = nn.ModuleList(
            conv_factory(hidden_channels, edge_hidden_channels) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(GraphNorm(hidden_channels) for _ in range(num_layers))
        self.grus = nn.ModuleList(
            nn.GRUCell(hidden_channels, hidden_channels) for _ in range(num_layers)
        )
        self.fc = nn.Linear(hidden_channels, hidden_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        for conv, norm, gru in zip(self.convs, self.norms, self.grus, strict=True):
            m = self.dropout(self.act(norm(conv(h, edge_index, e), batch)))
            h = gru(m, h)
        return self.act(self.fc(h))


class PNAEncoder(_GRUMessagePassingStack):
    """PNAConv with per-layer ``GraphNorm`` + ``GRUCell`` state updater.

    ``deg`` is the in-degree histogram of the training graphs — required by
    ``PNAConv`` to size its log-degree scaler buffer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        edge_hidden_channels: int = 128,
        num_layers: int = 6,
        deg: torch.Tensor | list[int] | None = None,
        aggregators: tuple[str, ...] = ("mean", "max", "min", "std"),
        scalers: tuple[str, ...] = ("identity", "amplification", "attenuation"),
        towers: int = 1,
        dropout: float = 0.1,
        pre_layers: int = 1,
        post_layers: int = 1,
    ) -> None:
        deg_t = (
            torch.tensor([0, 1, 5, 5, 1], dtype=torch.long)
            if deg is None
            else torch.as_tensor(deg, dtype=torch.long)
        )

        def make_conv(hidden: int, edge_hidden: int) -> nn.Module:
            return PNAConv(
                in_channels=hidden,
                out_channels=hidden,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg_t,
                edge_dim=edge_hidden,
                towers=towers,
                pre_layers=pre_layers,
                post_layers=post_layers,
            )

        super().__init__(
            make_conv,
            in_channels,
            hidden_channels,
            edge_dim,
            edge_hidden_channels,
            num_layers,
            dropout,
        )


class MPNNEncoder(_GRUMessagePassingStack):
    """Independent ``NNConv`` per layer + ``GraphNorm`` + ``GRUCell`` — PNA ablation.

    Same backbone as ``PNAEncoder``; only the conv operator changes, so the
    ablation isolates the conv choice and nothing else.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        edge_hidden_channels: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        def make_conv(hidden: int, edge_hidden: int) -> nn.Module:
            edge_net = nn.Sequential(
                nn.Linear(edge_hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden * hidden),
            )
            return NNConv(hidden, hidden, nn=edge_net, aggr="mean")

        super().__init__(
            make_conv,
            in_channels,
            hidden_channels,
            edge_dim,
            edge_hidden_channels,
            num_layers,
            dropout,
        )
