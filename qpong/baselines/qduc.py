"""QDUC baseline: quantum distributed unit commitment (Nikmehr, Zhang & Bragin, 2022).

ADMM decomposition between
  * a binary commitment subproblem, posed as a QUBO and solved with QAOA
    (depth ``qaoa_p``, exact statevector, COBYLA), and
  * a continuous economic-dispatch subproblem (quadratic costs, box limits),
coupled through the consensus P_i = P_hat_i u_i with scaled duals.
The resulting commitment and dispatch are issued directly: QDUC has no
physics-informed prediction, AC verification or governance layer.
"""
from __future__ import annotations

import time

import numpy as np

from ..stage2_quantum.qaoa import QAOA
from ..stage2_quantum.qubo import build_qubo, hard_feasible
from ..ucproblem import make_uc_problem
from .common import default_vg, economic_dispatch, planned_state_pf, record_issue


class QDUC:
    label = "QDUC"

    def __init__(self, grid, cfg, rng=None):
        self.grid, self.cfg = grid, cfg
        self.b = cfg["baselines"]["qduc"]
        self.rng = rng or np.random.default_rng(0)

    def _admm(self, prob, net_load):
        """ADMM; the binary step is distributed over subsystems of at most ``group_size``
        units, each solved by QAOA with the other subsystems held at their current values."""
        grid, b = self.grid, self.b
        n = grid.n_enc
        lam = self.cfg["stage2"]["penalties"]
        rho = b["admm_rho"]
        pbar = np.clip(prob.target_loading * prob.pmax, prob.pmin, prob.pmax)
        q = build_qubo(prob, lam)
        groups = np.array_split(np.arange(n), int(np.ceil(n / b.get("group_size", 7))))
        u = np.ones(n)
        p = pbar.copy()
        dual = np.zeros(n)
        for _ in range(b["admm_iters"]):
            u_old = u.copy()
            for G in groups:
                k = len(G)
                loc = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(float)
                Y = np.tile(u, (1 << k, 1))
                Y[:, G] = loc
                r = (p + dual)[None, :] - pbar[None, :] * Y
                E = q.energy(Y) + rho / 2 * np.sum((r / np.maximum(pbar, 1e-6)) ** 2, axis=1)
                _, pr = QAOA(E, k, b["qaoa_p"]).optimize(b["qaoa_maxiter"], self.rng)
                u = Y[int(np.argmax(pr))].copy()
            if not hard_feasible(prob, u[None, :], q)[0]:
                u = np.maximum(u, u_old)
            pg = economic_dispatch(grid, grid.full_commitment(u), net_load)
            p = pg[grid.encoded]
            dual = dual + (p - pbar * u)
            if np.array_equal(u, u_old):
                break
        return u

    def run_day(self, sc, intervals=None):
        grid, cfg = self.grid, self.cfg
        out, u_prev = [], None
        up = down = None
        w = np.array([1.0, 0.0, 0.0])
        for t in (intervals if intervals is not None else range(sc.T)):
            t0 = time.perf_counter()
            pd, ppv = sc.pd_id[t], sc.ppv_id[t]
            prob = make_uc_problem(grid, cfg, w, pd, ppv, pd, sc.co2_t_per_mwh,
                                   None if u_prev is None else u_prev[grid.encoded], up, down)
            net = pd.sum() - ppv.sum()
            u_enc = self._admm(prob, net)
            u = grid.full_commitment(u_enc)
            pg = economic_dispatch(grid, u, net)
            vg = default_vg(grid)
            el = time.perf_counter() - t0
            plan = planned_state_pf(grid, pd, sc.qd[t], ppv, u, pg, vg)
            out.append(record_issue(self.label, grid, sc, t, u, pg, vg, u_prev, {"solve": el}, planned=plan))
            if u_prev is None:
                up = np.full(grid.n_enc, 99)
                down = np.full(grid.n_enc, 99)
            else:
                pe = u_prev[grid.encoded]
                up = np.where(u_enc > 0.5, np.where(pe > 0.5, up + 1, 1), 0)
                down = np.where(u_enc < 0.5, np.where(pe < 0.5, down + 1, 1), 0)
            u_prev = u
        return out
