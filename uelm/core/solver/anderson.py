"""Anderson acceleration placeholder."""

from __future__ import annotations

import torch


def anderson_solve(z0: torch.Tensor, gradF, cfg=None) -> torch.Tensor:
    z = z0
    lr = 0.1
    for _ in range(3):
        z = z - lr * gradF(z)
    return z
