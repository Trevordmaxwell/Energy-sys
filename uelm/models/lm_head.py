"""Language modeling head."""

from __future__ import annotations

import torch


class LMHead(torch.nn.Module):
    def __init__(self, d: int, vocab_size: int = 50257, tie_weights: bool = False, embedding_layer=None) -> None:
        super().__init__()
        if tie_weights and embedding_layer is not None:
            self.proj = torch.nn.Linear(d, vocab_size, bias=False)
            self.proj.weight = embedding_layer.weight
        else:
            self.proj = torch.nn.Linear(d, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)
