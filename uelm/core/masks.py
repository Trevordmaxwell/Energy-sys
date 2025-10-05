"""Mask utilities."""

from __future__ import annotations

import torch


def causal_mask(length: int) -> torch.Tensor:
    return torch.tril(torch.ones(length, length))
