"""Classical solvers for the encoded QUBO.

* ``exact_enumeration`` - exhaustive ground truth (2^N states, vectorized).
* ``gurobi_miqp``       - optional Gurobi MIQP cross-check (Section VI-C).
* ``simulated_annealing`` - ablation "Classical (SA)" (2000 sweeps, 10 restarts)
  and best-known reference for N = 25, 30 together with ``tabu_search``.
"""
from __future__ import annotations

import numpy as np


def exact_enumeration(qubo, k: int = 1):
    E = qubo.energy_all()
    order = np.argsort(E, kind="stable")[:k]
    Y = ((order[:, None] >> np.arange(qubo.n)) & 1).astype(np.int8)
    return Y, E[order], E


def gurobi_miqp(qubo, time_limit=600, mip_gap=0.0):
    import gurobipy as gp

    m = gp.Model("qubo")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    y = m.addMVar(qubo.n, vtype=gp.GRB.BINARY)
    m.setObjective(y @ qubo.Q @ y + qubo.c @ y + qubo.c0, gp.GRB.MINIMIZE)
    m.optimize()
    return np.round(y.X).astype(np.int8), float(m.ObjVal), int(m.Status)


def simulated_annealing(qubo, sweeps=2000, restarts=10, rng=None, T0=None, T1=None):
    """Single-flip Metropolis annealing, geometric schedule; restarts run as a batch."""
    rng = rng or np.random.default_rng()
    n, R = qubo.n, restarts
    Q2 = 2 * qubo.Q
    scale = np.abs(qubo.c).max() + np.abs(Q2).sum(axis=1).max() + 1e-12
    temps = np.geomspace(T0 or scale, T1 or scale * 1e-4, sweeps)
    Y = rng.integers(0, 2, (R, n)).astype(float)
    F = qubo.c[None, :] + Y @ Q2.T          # dE of flipping 0->1 for each bit
    np.arange(R)
    for T in temps:
        for k in rng.permutation(n):
            dE = F[:, k] * (1 - 2 * Y[:, k])
            acc = (dE <= 0) | (rng.random(R) < np.exp(-np.clip(dE, 0, None) / T))
            if acc.any():
                step = (1 - 2 * Y[acc, k])
                Y[acc, k] += step
                F[acc] += step[:, None] * Q2[:, k][None, :]
    E = qubo.energy(Y)
    j = int(np.argmin(E))
    return Y[j].astype(np.int8), float(E[j])


def tabu_search(qubo, iters=5000, tenure=None, restarts=10, rng=None):
    """1-flip tabu search with aspiration; restarts run as a batch."""
    rng = rng or np.random.default_rng()
    n, R = qubo.n, restarts
    tenure = tenure or max(3, n // 4)
    Q2 = 2 * qubo.Q
    Y = rng.integers(0, 2, (R, n)).astype(float)
    F = qubo.c[None, :] + Y @ Q2.T
    e = qubo.energy(Y)
    best_e, best_Y = e.copy(), Y.copy()
    tabu = np.zeros((R, n), int)
    rows = np.arange(R)
    for it in range(iters):
        dE = F * (1 - 2 * Y)
        allowed = (tabu <= it) | (e[:, None] + dE < best_e[:, None])
        k = np.argmin(np.where(allowed, dE, np.inf), axis=1)
        step = 1 - 2 * Y[rows, k]
        Y[rows, k] += step
        e = e + dE[rows, k]
        F += step[:, None] * Q2[:, k].T
        tabu[rows, k] = it + tenure
        imp = e < best_e
        best_e[imp] = e[imp]
        best_Y[imp] = Y[imp]
    j = int(np.argmin(best_e))
    return best_Y[j].astype(np.int8), float(qubo.energy(best_Y[j])[0])
