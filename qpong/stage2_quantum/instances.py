"""Synthetic commitment-only UC-QUBO instances (Section V-A), independent of any bus system.

Used for the scalability sweep (N_enc = 5..30, Table V) and for validation
against exact ground truth (N_enc = 4..20, Table VI). Unit data are drawn
from ranges typical of the IEEE test cases; the demand is set to a fraction
of the installed capacity. Instances are deterministic in (N, seed).
"""
from __future__ import annotations

import numpy as np

from .qubo import UCProblem


def synthetic_uc(n: int, seed: int = 0, load_fraction: float = 0.55, weights=(0.5, 0.2, 0.3),
                 target_loading: float = 0.75, pmin_fraction: float = 0.3) -> UCProblem:
    rng = np.random.default_rng(10_000 + 97 * n + seed)
    pmax = rng.uniform(50, 500, n).round(0)
    b = rng.uniform(10, 45, n)
    a = rng.uniform(0.002, 0.06, n)
    c = 0.1 * b * pmax
    em = np.full(n, 0.4)
    rho = rng.uniform(0.05, 1.0, n)
    D = load_fraction * pmax.sum()
    return UCProblem(pmax=pmax, pmin=pmin_fraction * pmax, a=a, b=b, c=c, startup=b * pmax,
                     emission=em, rho=rho, demand=D, demand_hat=D, res=0.0,
                     weights=np.asarray(weights, float), target_loading=target_loading)
