"""Fuzzy Shapley allocation (eq. 5) - exact enumeration and shared permutations.

Coalitions are bitmasks over participants 0..n-1. ``value_fn(masks)`` must
return arrays (m, s) of GFN means and spreads for a batch of masks; it is
called once on all distinct coalitions needed (cached by bitmask).

Means add linearly; spreads combine in quadrature (independence, eq. 4):
  phi_i = sum_S w_S [v(S u {i}) - v(S)]
  m_phi = sum_S w_S (m_{S+i} - m_S)
  s_phi = sqrt( sum_S w_S^2 (s_{S+i}^2 + s_S^2) )
For M shared permutations every participant uses the same sampled orders,
so the means telescope and efficiency holds exactly (Proposition 2).
"""
from __future__ import annotations

from math import factorial

import numpy as np


def exact_shapley(n: int, value_fn):
    masks = np.arange(1 << n, dtype=np.int64)
    m, s = value_fn(masks)
    m = np.asarray(m, float)
    s = np.asarray(s, float)
    size = np.array([bin(x).count("1") for x in masks])
    w_by_size = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
    mphi = np.zeros(n)
    var = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        S = masks[(masks & bit) == 0]
        w = w_by_size[size[S]]
        Si = S | bit
        mphi[i] = np.sum(w * (m[Si] - m[S]))
        var[i] = np.sum(w ** 2 * (s[Si] ** 2 + s[S] ** 2))
    return mphi, np.sqrt(var), {"n_coalitions": len(masks), "m": m, "s": s}


def permutation_shapley(n: int, value_fn, M: int = 1000, rng=None):
    rng = rng or np.random.default_rng()
    perms = np.stack([rng.permutation(n) for _ in range(M)])
    prefix = np.zeros((M, n + 1), dtype=np.int64)
    for k in range(n):
        prefix[:, k + 1] = prefix[:, k] | (np.int64(1) << perms[:, k].astype(np.int64))
    uniq, inv = np.unique(prefix.ravel(), return_inverse=True)
    m_u, s_u = value_fn(uniq)
    m_u, s_u = np.asarray(m_u, float), np.asarray(s_u, float)
    mv = m_u[inv].reshape(M, n + 1)
    sv = s_u[inv].reshape(M, n + 1)
    mphi = np.zeros(n)
    var = np.zeros(n)
    for k in range(n):
        who = perms[:, k]
        dm = mv[:, k + 1] - mv[:, k]
        dv = (sv[:, k + 1] ** 2 + sv[:, k] ** 2) / M ** 2
        np.add.at(mphi, who, dm / M)
        np.add.at(var, who, dv)
    return mphi, np.sqrt(var), {"n_coalitions": int(len(uniq)), "n_prefix": int(M * n)}


def shapley(n, value_fn, exact_max_n=10, M=1000, rng=None):
    if n <= exact_max_n:
        return exact_shapley(n, value_fn)
    return permutation_shapley(n, value_fn, M, rng)


def coalition_efficiency_error(mphi, m_grand, m_empty=0.0):
    """E_coal of eq. (29)."""
    return float(abs(np.sum(mphi) - (m_grand - m_empty)))
