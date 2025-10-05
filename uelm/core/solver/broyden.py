"""Broyden solver placeholder."""

from __future__ import annotations

import torch


def broyden_solve(z0: torch.Tensor, gradF, cfg=None) -> torch.Tensor:
    del gradF, cfg
    return z0
