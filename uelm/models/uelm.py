"""Unified Equilibrium Language Model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uelm.core.cde import build_piecewise_linear_path
from uelm.core.energy import total_energy
from uelm.core.hopfield import HopfieldMemory
from uelm.core.solver.anderson import anderson_solve
from uelm.models.embeddings import Embeddings
from uelm.models.lm_head import LMHead


@dataclass
class UELMConfig:
    d: int
    dx: int
    K: int
    beta_init: float
    mu: float
    lambda_h: float
    lambda_norm: float
    causal_band: int
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    solver: dict | None = None


class UELM(torch.nn.Module):
    def __init__(self, cfg: UELMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg.d)
        self.mem = HopfieldMemory(cfg.d, cfg.K, beta_init=cfg.beta_init)
        self.lm_head = LMHead(cfg.d)
        self.A = torch.nn.Linear(cfg.d, cfg.d, bias=False)

    @torch.no_grad()
    def _warm_start(self, E: torch.Tensor) -> torch.Tensor:
        return self.A(E)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        E = self.embed(tokens)
        X = build_piecewise_linear_path(E)
        z0 = self._warm_start(E).detach().requires_grad_(True)

        def gradF(z: torch.Tensor) -> torch.Tensor:
            F = total_energy(z, E, X, self.mem, self.cfg)
            (g,) = torch.autograd.grad(F, z, create_graph=False)
            return g

        z_star = anderson_solve(z0, gradF, self.cfg.solver)
        logits = self.lm_head(z_star)
        return logits
