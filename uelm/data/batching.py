"""Batching helpers."""

from __future__ import annotations

import torch


def make_batches(tokens: torch.Tensor, seq_len: int) -> torch.Tensor:
    total = tokens.size(0)
    trimmed = total // seq_len * seq_len
    return tokens[:trimmed].view(-1, seq_len)
