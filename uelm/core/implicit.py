"""Implicit differentiation helpers (placeholder)."""

from __future__ import annotations

import torch


def solve_vjp(z_star: torch.Tensor, grad_outputs: torch.Tensor) -> torch.Tensor:
    del z_star
    return grad_outputs
