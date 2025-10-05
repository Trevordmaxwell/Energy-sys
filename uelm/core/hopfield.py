"""Modern Hopfield memory placeholder."""

from __future__ import annotations

import torch


class HopfieldMemory(torch.nn.Module):
    """CPU-friendly Hopfield memory stub."""

    def __init__(self, d: int, K: int, beta_init: float = 0.05) -> None:
        super().__init__()
        self.patterns = torch.nn.Parameter(torch.randn(K, d) * 0.02)
        self.beta = torch.nn.Parameter(torch.tensor(beta_init))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query, self.patterns.t())
        weights = torch.softmax(self.beta * scores, dim=-1)
        return torch.matmul(weights, self.patterns)
