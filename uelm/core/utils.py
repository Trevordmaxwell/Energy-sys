"""Utility helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SolverConfig:
    max_iter: int = 15
    tol: float = 1.0e-3
