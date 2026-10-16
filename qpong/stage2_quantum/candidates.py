"""Candidate set U_t and pre-screen ranking J_cand (eq. 14)."""
from __future__ import annotations

import numpy as np

from .qubo import hard_feasible
from .samplers import decode_indices


def top_candidates(counts: dict, n: int, max_candidates: int = 100):
    """N_cand = min(100, 2^N) most frequent distinct bitstrings (fewer if fewer were observed)."""
    k = min(max_candidates, 1 << n)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    idx = [i for i, _ in items]
    freq = np.array([c for _, c in items], float)
    return decode_indices(idx, n), freq


def j_cand(prob, qubo, Y: np.ndarray, lam: dict, omega: dict) -> np.ndarray:
    """J_cand(y) = E_t(y) + lambda_F (omega_P |r_P| + omega_G N_viol_G + omega_D |D - D_hat|)."""
    E = qubo.energy(Y)
    P = Y @ qubo.P_map.T
    U = Y @ qubo.U_map.T
    D = max(prob.demand, 1e-6)
    r_p = (P.sum(axis=1) + prob.res + prob.p_fixed - prob.demand) / D
    cap = U @ prob.pmax + prob.p_fixed_max
    nl = prob.demand - prob.res
    n_g = ((cap < nl * (1 + prob.reserve)).astype(int)
           + ((U @ prob.pmin + prob.p_fixed) > max(nl, 0)).astype(int))
    d_dev = abs(prob.demand - prob.demand_hat) / D
    return E + lam["lambda_F"] * (omega["omega_P"] * np.abs(r_p) + omega["omega_G"] * n_g + omega["omega_D"] * d_dev)


def rank_candidates(prob, qubo, counts: dict, cfg: dict):
    s2 = cfg["stage2"]
    Y, freq = top_candidates(counts, qubo.n, s2["max_candidates"])
    J = j_cand(prob, qubo, Y, s2["penalties"], s2["prescreen_weights"])
    order = np.argsort(J, kind="stable")
    return Y[order], J[order], freq[order]


def shot_feasible_share(prob, qubo, counts: dict) -> tuple[int, int]:
    """(S_feas, S_shots): shots whose decoded bitstring satisfies the hard constraints."""
    keys = list(counts.keys())
    Y = decode_indices(keys, qubo.n)
    ok = hard_feasible(prob, Y, qubo)
    c = np.array([counts[k] for k in keys])
    return int(c[ok].sum()), int(c.sum())
