"""Numerical operations."""

from __future__ import annotations

import torch


def stable_logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    return max_val + torch.log(torch.sum(torch.exp(x - max_val), dim=dim, keepdim=True))
