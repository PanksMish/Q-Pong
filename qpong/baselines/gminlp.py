"""G-MINLP reference: mixed-integer AC unit commitment with the same objective,
commitment variables and constraints as Q-PONG (reference for the optimality gap).

Back-ends
---------
``gurobi``   : Gurobi (>= 11) nonconvex MIQCP in rectangular voltage coordinates
               (V = e + j f): bilinear AC power balance, |V|^2 bounds, apparent-power
               branch limits via auxiliary flow variables, binary commitment with
               Pmin/Pmax/Qmin/Qmax, start-up and min up/down state, and the
               scalarized objective of eq. (1) with the interval's FBWM weights.
               Solved to MIPGap 1e-4 within ``time_limit_s`` (600 s); if optimality
               is not certified, the best incumbent is used (Section V-D).
``enumerate``: exhaustive search over the 2^N encoded commitments that satisfy the
               hard UC constraints, each dispatched by AC-OPF (MATPOWER MIPS) and
               scored by the same objective. Exact for the encoded space when N is
               small (30/57-bus); for N > ``max_exhaustive`` the commitments are
               screened by the QUBO surrogate and the best ``enumerate_top_k`` are
               solved by AC-OPF (NOT certified - reported as such).
``auto``     : gurobi if gurobipy and a licence are available, else enumerate.
"""
from __future__ import annotations

import time

import numpy as np

from ..grid.acopf import solve_acopf
from ..objective import scalarized
from ..stage1_fbwm import FBWM, context_vector
from ..stage2_quantum.qubo import build_qubo, hard_feasible
from ..ucproblem import make_uc_problem
from .common import record_issue


def _have_gurobi():
    try:
        import gurobipy as gp

        gp.Model().dispose()
        return True
    except Exception:
        return False


class GMINLP:
    label = "G-MINLP"

    def __init__(self, grid, cfg, rng=None):
        self.grid, self.cfg = grid, cfg
        b = cfg["baselines"]["gminlp"]
        self.backend = b["backend"]
        if self.backend == "auto":
            self.backend = "gurobi" if _have_gurobi() else "enumerate"
        self.fbwm = FBWM(cfg)
        self.dt = cfg["data"]["interval_minutes"] / 60

    # ---------------------------------------------------------------- enumerate
    def _enumerate(self, prob, pd, qd, ppv, w, u_prev, co2, rho_units):
        grid, b = self.grid, self.cfg["baselines"]["gminlp"]
        n = grid.n_enc
        idx = np.arange(1 << n, dtype=np.int64)
        Y = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)
        qubo = build_qubo(prob, self.cfg["stage2"]["penalties"])
        ok = hard_feasible(prob, Y, qubo)
        Y = Y[ok]
        certified = n <= b.get("max_exhaustive", 10)
        if not certified:          # screen by the QUBO surrogate, solve in growing batches
            Y = Y[np.argsort(qubo.energy(Y), kind="stable")]
        best = (np.inf, None, None)
        t0 = time.perf_counter()
        k = len(Y) if certified else b["enumerate_top_k"]
        start = 0
        while start < len(Y):
            for y in Y[start:k]:
                if time.perf_counter() - t0 > b["time_limit_s"]:
                    return best, False
                u = grid.full_commitment(y)
                r = solve_acopf(grid, pd, qd, ppv, u)
                if not r.success:
                    continue
                F = scalarized(grid, self.cfg, w, u, r.pg, u_prev, co2, rho_units)
                if F < best[0]:
                    best = (F, u, r)
            if best[1] is not None or certified:
                break
            start, k = k, min(len(Y), 2 * k)      # nothing feasible yet: widen the screen
        return best, certified

    # ------------------------------------------------------------------ gurobi
    def _gurobi(self, pd, qd, ppv, w, u_prev, up, down, co2, rho_units):
        import gurobipy as gp
        from gurobipy import GRB

        grid, cfg = self.grid, self.cfg
        b = cfg["baselines"]["gminlp"]
        g = cfg["grid"]
        U = grid.units
        nb, ng, bm = grid.n_bus, grid.n_units, grid.base_mva
        Gm, Bm = grid.G, grid.B
        Yf, Yt = grid.Yf.toarray(), grid.Yt.toarray()
        from ..objective import normalizers

        n_cost, n_carb, n_grid = normalizers(grid, cfg, co2, rho_units)
        m = gp.Model("gminlp")
        m.Params.OutputFlag = 0
        m.Params.NonConvex = 2
        m.Params.TimeLimit = b["time_limit_s"]
        m.Params.MIPGap = b["mip_gap"]
        e = m.addVars(nb, lb=-1.2, ub=1.2, name="e")
        f = m.addVars(nb, lb=-1.2, ub=1.2, name="f")
        u = m.addVars(ng, vtype=GRB.BINARY, name="u")
        p = m.addVars(ng, lb=-GRB.INFINITY, name="p")       # pu
        q = m.addVars(ng, lb=-GRB.INFINITY, name="q")
        enc = set(grid.encoded.tolist())
        for i in range(ng):
            if i not in enc:
                m.addConstr(u[i] == 1)
            at_ref = U.bus.iloc[i] == grid.slack_bus
            m.addConstr(p[i] >= U.pmin.iloc[i] / bm * u[i])
            m.addConstr(p[i] <= U.pmax.iloc[i] / bm * u[i])
            qlo, qhi = U.qmin.iloc[i] / bm, U.qmax.iloc[i] / bm
            if at_ref:   # a decommitted reference unit stays as a condenser
                m.addConstr(q[i] >= qlo)
                m.addConstr(q[i] <= qhi)
            else:
                m.addConstr(q[i] >= qlo * u[i])
                m.addConstr(q[i] <= qhi * u[i])
            if u_prev is not None and i in enc:
                k = list(grid.encoded).index(i)
                if u_prev[i] > 0.5 and up is not None and up[k] < g["min_up_intervals"]:
                    m.addConstr(u[i] == 1)
                if u_prev[i] < 0.5 and down is not None and down[k] < g["min_down_intervals"]:
                    m.addConstr(u[i] == 0)
        m.addConstr(f[grid.slack_bus] == 0)
        m.addConstr(e[grid.slack_bus] >= 0)
        for k in range(nb):
            m.addQConstr(e[k] * e[k] + f[k] * f[k] >= grid.vmin[k] ** 2)
            m.addQConstr(e[k] * e[k] + f[k] * f[k] <= grid.vmax[k] ** 2)
        C = grid.unit_bus_matrix()
        for k in range(nb):
            nz = np.nonzero(np.abs(Gm[k]) + np.abs(Bm[k]))[0]
            # P_k = e_k sum_j (G e_j - B f_j) + f_k sum_j (G f_j + B e_j)
            Pk = gp.QuadExpr()
            Qk = gp.QuadExpr()
            for j in nz:
                Pk += Gm[k, j] * (e[k] * e[j] + f[k] * f[j]) + Bm[k, j] * (f[k] * e[j] - e[k] * f[j])
                Qk += Gm[k, j] * (f[k] * e[j] - e[k] * f[j]) - Bm[k, j] * (e[k] * e[j] + f[k] * f[j])
            gens = np.nonzero(C[k])[0]
            m.addQConstr(gp.quicksum(p[i] for i in gens) - (pd[k] - ppv[k]) / bm == Pk)
            m.addQConstr(gp.quicksum(q[i] for i in gens) - qd[k] / bm == Qk)
        for l in range(len(grid.f_bus)):
            if not np.isfinite(grid.s_max[l]):
                continue
            for Ymat, bus in ((Yf, grid.f_bus[l]), (Yt, grid.t_bus[l])):
                row = Ymat[l]
                nz = np.nonzero(np.abs(row))[0]
                # I = sum_j y_j V_j ; S = V_bus conj(I)
                ir = gp.LinExpr()
                ii = gp.LinExpr()
                for j in nz:
                    ir += row[j].real * e[j] - row[j].imag * f[j]
                    ii += row[j].real * f[j] + row[j].imag * e[j]
                Ir = m.addVar(lb=-GRB.INFINITY)
                Ii = m.addVar(lb=-GRB.INFINITY)
                m.addConstr(Ir == ir)
                m.addConstr(Ii == ii)
                ps = m.addVar(lb=-GRB.INFINITY)
                qs = m.addVar(lb=-GRB.INFINITY)
                m.addQConstr(ps == e[bus] * Ir + f[bus] * Ii)
                m.addQConstr(qs == f[bus] * Ir - e[bus] * Ii)
                m.addQConstr(ps * ps + qs * qs <= grid.s_max[l] ** 2)
        dt = self.dt
        cost = gp.quicksum(U.a.iloc[i] * (bm * p[i]) * (bm * p[i]) + U.b.iloc[i] * bm * p[i] + U.c.iloc[i] * u[i]
                           for i in range(ng)) * dt
        if u_prev is not None:
            cost += gp.quicksum(U.startup.iloc[i] * u[i] * (1 - u_prev[i]) for i in enc)
        carb = gp.quicksum(bm * p[i] for i in range(ng)) * dt * co2
        sec = gp.quicksum(rho_units[i] * bm * p[i] for i in range(ng))
        m.setObjective(w[0] * cost / n_cost + w[1] * carb / n_carb + w[2] * sec / n_grid, GRB.MINIMIZE)
        m.optimize()
        if m.SolCount == 0:
            return None, False
        uu = np.round([u[i].X for i in range(ng)])
        certified = m.Status == GRB.OPTIMAL
        # AC-OPF polish on the chosen commitment gives MATPOWER-consistent set-points
        r = solve_acopf(grid, pd, qd, ppv, uu)
        return (uu, r), certified

    # --------------------------------------------------------------------- day
    def run_day(self, sc, intervals=None):
        grid, cfg = self.grid, self.cfg
        out = []
        u_prev = None
        up = down = None
        vm = loading = None
        p_fix = 0.0
        for t in (intervals if intervals is not None else range(sc.T)):
            pd, qd, ppv = sc.pd[t], sc.qd[t], sc.ppv[t]
            t0 = time.perf_counter()
            s = context_vector(cfg, ppv, sc.ppv_id[t], sc.price_rt[t], sc.price_ref, sc.freq[t], vm, 1.0, loading)
            w = self.fbwm(s)
            rho_units = grid.security_sensitivity(loading)
            prob = make_uc_problem(grid, cfg, w, pd, ppv, sc.pd_id[t], sc.co2_t_per_mwh,
                                   None if u_prev is None else u_prev[grid.encoded], up, down, p_fix, loading)
            if self.backend == "gurobi":
                sol, cert = self._gurobi(pd, qd, ppv, w, u_prev, up, down, sc.co2_t_per_mwh, rho_units)
                u, r = sol if sol else (np.ones(grid.n_units), None)
            else:
                (F, u, r), cert = self._enumerate(prob, pd, qd, ppv, w, u_prev, sc.co2_t_per_mwh, rho_units)
            el = time.perf_counter() - t0
            ok = r is not None and r.success
            rec = record_issue(self.label, grid, sc, t, u if ok else np.ones(grid.n_units),
                               r.pg if ok else None, r.vg if ok else None, u_prev, {"solve": el},
                               planned=(r.vm, r.va, r.pg, r.qg) if ok else None,
                               extra={"certified": bool(cert), "backend": self.backend})
            out.append(rec)
            if ok:
                ue = u[grid.encoded]
                if u_prev is None:
                    up = np.full(grid.n_enc, 99)
                    down = np.full(grid.n_enc, 99)
                else:
                    pe = u_prev[grid.encoded]
                    up = np.where(ue > 0.5, np.where(pe > 0.5, up + 1, 1), 0)
                    down = np.where(ue < 0.5, np.where(pe < 0.5, down + 1, 1), 0)
                u_prev = u
                vm = r.vm
                from ..grid.acpf import branch_loading

                loading = branch_loading(grid, r.vm, r.va)
                if len(grid.fixed):
                    p_fix = float(r.pg[grid.fixed].sum())
        return out
