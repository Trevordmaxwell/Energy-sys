"""Energy function definition for UELM."""

from __future__ import annotations

import torch


def total_energy(z: torch.Tensor, E: torch.Tensor, X: torch.Tensor, memory, cfg) -> torch.Tensor:
    """Placeholder energy function."""
    del E, X, memory, cfg
    return (z ** 2).sum()
