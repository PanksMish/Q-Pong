"""Reduced UC encoding: weighted objective (10), penalized QUBO (11), Ising map (12-13).

Encoding (eq. 9): one commitment bit per encoded unit/zone, plus optional
finite-resolution dispatch bits P_i = Pmin_i u_i + sum_k dP_ik b_ik.
With commitment bits only (K_i = 0, the setting of the reported runs), an
encoded unit that is committed is evaluated at its nominal loading
P_bar_i = clip(target_loading * Pmax_i, Pmin_i, Pmax_i) (MODELING CHOICE:
the paper does not state how P_i is set when K_i = 0). Every power term is
therefore linear in the bits, and E(y) = y^T Q y + c^T y + c0 is exact.

Normalization: each criterion is divided by its value with every encoded
unit committed (f_min = 0), i.e. the min-max scaling of eq. (1).
Power-balance terms are scaled by the net demand so the penalty is
dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class UCProblem:
    pmax: np.ndarray
    pmin: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    startup: np.ndarray
    emission: np.ndarray            # t/MWh per encoded unit
    rho: np.ndarray                 # network-security sensitivity per unit (R_t proxy)
    demand: float                   # D_t (MW)
    demand_hat: float               # forecast D_hat_t (MW)
    res: float                      # P_RES_t (MW)
    p_fixed: float = 0.0            # output of classically fixed units (MW)
    p_fixed_max: float = 0.0
    weights: np.ndarray = field(default_factory=lambda: np.ones(3) / 3)
    u_prev: np.ndarray | None = None
    up_time: np.ndarray | None = None     # intervals since last switch (when on)
    down_time: np.ndarray | None = None   # (when off)
    min_up: int = 4
    min_down: int = 4
    target_loading: float = 0.75
    reserve: float = 0.10
    dispatch_bits: np.ndarray | None = None   # K_i per unit (None/0 = commitment only)
    dt_h: float = 0.25

    @property
    def n_units(self) -> int:
        return len(self.pmax)

    @property
    def net_load(self) -> float:
        return self.demand - self.res - self.p_fixed


@dataclass
class QUBO:
    Q: np.ndarray            # symmetric, zero diagonal
    c: np.ndarray
    c0: float
    P_map: np.ndarray        # (n_units, N) : P = P_map @ y
    U_map: np.ndarray        # (n_units, N) : u = U_map @ y
    n: int
    meta: dict = field(default_factory=dict)

    def energy(self, Y: np.ndarray) -> np.ndarray:
        """E(y) for a batch of bit vectors Y (B, N) or a single vector."""
        Y = np.atleast_2d(Y).astype(float)
        return self.c0 + Y @ self.c + np.einsum("bi,ij,bj->b", Y, self.Q, Y)

    def energy_all(self, chunk: int = 1 << 18) -> np.ndarray:
        """Energies of all 2^N states (index = little-endian bit order, bit k = y_k)."""
        N = self.n
        out = np.empty(1 << N)
        for s in range(0, 1 << N, chunk):
            idx = np.arange(s, min(s + chunk, 1 << N), dtype=np.int64)
            Y = ((idx[:, None] >> np.arange(N)) & 1).astype(float)
            out[s:s + len(idx)] = self.energy(Y)
        return out


def _layout(prob: UCProblem):
    """Bit layout: commitment bits first (one per unit), then dispatch bits."""
    n = prob.n_units
    K = np.zeros(n, int) if prob.dispatch_bits is None else np.asarray(prob.dispatch_bits, int)
    N = n + int(K.sum())
    P_map = np.zeros((n, N))
    U_map = np.zeros((n, N))
    U_map[np.arange(n), np.arange(n)] = 1.0
    pos = n
    pbar = np.clip(prob.target_loading * prob.pmax, prob.pmin, prob.pmax)
    bit_owner = list(range(n))
    for i in range(n):
        if K[i] == 0:
            P_map[i, i] = pbar[i]
        else:
            P_map[i, i] = prob.pmin[i]
            span = prob.pmax[i] - prob.pmin[i]
            steps = 2.0 ** np.arange(K[i])
            dP = span * steps / steps.sum()
            for k in range(K[i]):
                P_map[i, pos] = dP[k]
                bit_owner.append(i)
                pos += 1
    return N, P_map, U_map, np.array(bit_owner), K


def build_qubo(prob: UCProblem, lam: dict) -> QUBO:
    N, P_map, U_map, owner, K = _layout(prob)
    Q = np.zeros((N, N))
    c = np.zeros(N)
    c0 = 0.0
    w_cost, w_carbon, w_grid = prob.weights
    dt = prob.dt_h

    # ---- normalizations (value with all encoded units committed at nominal loading)
    p_all = P_map @ np.ones(N) if K.sum() == 0 else np.clip(prob.target_loading * prob.pmax, prob.pmin, prob.pmax)
    f_cost_max = np.sum(prob.a * p_all ** 2 + prob.b * p_all + prob.c) * dt + 1e-9
    f_carb_max = np.sum(prob.emission * p_all) * dt + 1e-9
    f_grid_max = np.sum(prob.rho * p_all) + 1e-9

    def add_quad_form(M, scale):
        """add scale * ||M y||^2 for rows of M (each row a linear form)."""
        nonlocal Q, c
        G = scale * (M.T @ M)
        c += np.diag(G)
        off = G - np.diag(np.diag(G))
        Q += off

    # ---- objective F_UC (10)
    # cost: a P^2 + b P + c u
    for i in range(prob.n_units):
        add_quad_form(P_map[i:i + 1], w_cost * prob.a[i] * dt / f_cost_max)
    c += w_cost * dt * (prob.b @ P_map + prob.c @ U_map) / f_cost_max
    c += w_carbon * dt * (prob.emission @ P_map) / f_carb_max
    c += w_grid * (prob.rho @ P_map) / f_grid_max

    # ---- power balance penalty  lam_P ((sum P + RES + P_fixed - D)/D)^2
    D = max(prob.demand, 1e-6)
    row = (P_map.sum(axis=0) / D)[None, :]
    k0 = (prob.res + prob.p_fixed - prob.demand) / D
    add_quad_form(row, lam["lambda_P"])
    c += lam["lambda_P"] * 2 * k0 * row[0]
    c0 += lam["lambda_P"] * k0 ** 2

    # ---- generator limits: dispatch bit of an uncommitted unit  lam_G b (1 - u)
    if K.sum() > 0:
        for j in range(prob.n_units, N):
            i = owner[j]
            scale = lam["lambda_G"] * (P_map[i, j] / max(prob.pmax[i], 1e-6)) ** 2
            c[j] += scale
            Q[i, j] -= scale / 2
            Q[j, i] -= scale / 2

    # ---- transitions Psi_i(u_t, u_{t-1}): start-up cost and min up/down time
    if prob.u_prev is not None:
        su_norm = f_cost_max
        big = 1.0
        for i in range(prob.n_units):
            if prob.u_prev[i] < 0.5:
                c[i] += lam["lambda_T"] * prob.startup[i] / su_norm
                if prob.down_time is not None and prob.down_time[i] < prob.min_down:
                    c[i] += lam["lambda_T"] * big * 10
            else:
                if prob.up_time is not None and prob.up_time[i] < prob.min_up:
                    c[i] -= lam["lambda_T"] * big * 10
                    c0 += lam["lambda_T"] * big * 10

    Q = (Q + Q.T) / 2
    return QUBO(Q=Q, c=c, c0=c0, P_map=P_map, U_map=U_map, n=N,
                meta={"f_cost_max": f_cost_max, "f_carb_max": f_carb_max, "f_grid_max": f_grid_max})


def qubo_to_ising(q: QUBO):
    """y_k = (1 - z_k)/2  ->  H = kappa I + sum h_k Z_k + sum_{k<l} J_kl Z_k Z_l (eq. 13)."""
    # y^T Q y = 2 sum_{k<l} Q_kl y_k y_l  and  y_k y_l = (1 - z_k - z_l + z_k z_l) / 4
    Qs = q.Q
    h = -q.c / 2 - Qs.sum(axis=1) / 2
    J = np.triu(Qs, 1) / 2
    kappa = q.c0 + q.c.sum() / 2 + np.triu(Qs, 1).sum() / 2
    return kappa, h, J


def ising_energy(kappa, h, J, Z: np.ndarray) -> np.ndarray:
    Z = np.atleast_2d(Z).astype(float)
    return kappa + Z @ h + np.einsum("bi,ij,bj->b", Z, J, Z)


def to_sparse_pauli_op(kappa, h, J):
    """Ising Hamiltonian as a qiskit SparsePauliOp (qubit k <-> bit y_k)."""
    from qiskit.quantum_info import SparsePauliOp

    N = len(h)
    terms = [("I" * N, kappa)]
    for k in range(N):
        if abs(h[k]) > 0:
            s = ["I"] * N
            s[N - 1 - k] = "Z"
            terms.append(("".join(s), h[k]))
    for k in range(N):
        for l in range(k + 1, N):
            if abs(J[k, l]) > 0:
                s = ["I"] * N
                s[N - 1 - k] = "Z"
                s[N - 1 - l] = "Z"
                terms.append(("".join(s), J[k, l]))
    return SparsePauliOp.from_list(terms).simplify()


def hard_feasible(prob: UCProblem, Y: np.ndarray, q: QUBO) -> np.ndarray:
    """Hard UC constraints of a decoded bit vector: adequacy with reserve,
    minimum-generation, and min up/down times."""
    Y = np.atleast_2d(Y)
    U = Y @ q.U_map.T
    cap = U @ prob.pmax + prob.p_fixed_max
    pmin = U @ prob.pmin + 0.0
    nl = prob.demand - prob.res
    ok = (cap >= nl * (1 + prob.reserve)) & (pmin + prob.p_fixed <= max(nl, 0.0) + 1e-9)
    if prob.u_prev is not None:
        if prob.up_time is not None:
            must_on = (prob.u_prev > 0.5) & (prob.up_time < prob.min_up)
            ok &= np.all(U[:, must_on] > 0.5, axis=1)
        if prob.down_time is not None:
            must_off = (prob.u_prev < 0.5) & (prob.down_time < prob.min_down)
            ok &= np.all(U[:, must_off] < 0.5, axis=1)
    return ok
