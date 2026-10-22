"""Farm-specific CPU networks with learned calendar/auction embeddings."""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Learn nonlinear weather interactions while preserving a direct hidden path."""

    def __init__(self, width, dropout):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width, width))

    def forward(self, value):
        return torch.nn.functional.gelu((value+self.body(value)) / 2**.5)


class TemporalBlock(nn.Module):
    """Mix only weather already known at the cutoff, including later forecast leads.

    The symmetric convolution is intentional: the entire chosen forecast curve
    is available at issue time. No future measured power or later vintage enters.
    Separate day/auction tensors prevent mixing delivery books or weather cycles.
    """

    def __init__(self, width, kernel, dilation, dropout):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(width, width, kernel,
            padding=dilation*(kernel-1)//2, dilation=dilation), nn.GELU(),
            nn.Dropout(dropout), nn.Conv1d(width, width, 1))

    def forward(self, value):
        return torch.nn.functional.gelu((value+self.body(value))/2**.5)


class PVNetwork(nn.Module):
    """Predict AC fraction without fitted DC, thermal, tracker or output parameters."""

    def __init__(self, input_features: int, candidate: dict, section: dict):
        super().__init__()
        self.architecture = candidate["architecture"]
        month_dim, auction_dim = section["month_embedding_dim"], section["auction_embedding_dim"]
        self.month_embedding = nn.Embedding(12, month_dim)
        self.auction_embedding = nn.Embedding(4, auction_dim)
        width, depth, dropout = candidate["width"], candidate["depth"], section["dropout"]
        self.stem = nn.Sequential(nn.Linear(input_features+month_dim+auction_dim, width), nn.GELU())
        if self.architecture == "mlp":
            self.body = nn.Sequential(*[layer for _ in range(depth-1) for layer in
                (nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))])
        elif self.architecture == "residual":
            self.body = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(depth)])
        elif self.architecture == "tcn":
            self.body = nn.Sequential(*[TemporalBlock(width, candidate["kernel_size"], 2**i, dropout)
                                        for i in range(depth)])
        else:
            raise ValueError("Unknown neural architecture")
        self.output = nn.Linear(width, 1)

    def forward(self, x, month, auction, daylight):
        embedded = torch.cat([x, self.month_embedding(month), self.auction_embedding(auction)], dim=-1)
        hidden = self.stem(embedded)
        if self.architecture == "tcn":
            hidden = self.body(hidden.transpose(1, 2)).transpose(1, 2)
        else:
            hidden = self.body(hidden)
        return torch.sigmoid(self.output(hidden).squeeze(-1))*daylight
