"""Embedding layers."""

from __future__ import annotations

import torch


class Embeddings(torch.nn.Module):
    def __init__(self, d: int, vocab_size: int = 50257) -> None:
        super().__init__()
        self.token = torch.nn.Embedding(vocab_size, d)
        self.pos = torch.nn.Embedding(2048, d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        return self.token(tokens) + self.pos(positions)
