"""Day-ahead baselines FBWM [Guo & Zhao 2017] and GFN-SV [Choucair et al. 2026].

Both solve one day-ahead mixed-integer linear UC with DC network constraints
(the commitment stage of G-MINLP without its AC constraints) on the
day-ahead forecasts, once per day; every interval is then dispatched by an
AC-OPF on the same day-ahead forecasts, and the set-points are applied in
real time. Their per-interval latency is the amortized share of the daily
UC plus one AC-OPF.

* FBWM   : weights from the (static) fuzzy BWM given the day's mean context;
           costs settled by deterministic Shapley allocation (sigma = 0).
* GFN-SV : fixed equal weights; Gaussian-fuzzy Shapley settlement of the fixed
           day-ahead schedule with spreads from day-ahead forecast errors.
"""
from __future__ import annotations

import time

import numpy as np

from ..grid.acopf import solve_acopf
from ..stage1_fbwm import FBWM, context_vector
from ..stage4_settlement.coalition import forecast_error_sd, masks_to_bits
from ..stage4_settlement.governance import relative_spread
from ..stage4_settlement.shapley import coalition_efficiency_error, shapley
from .common import default_vg, planned_state_pf, record_issue, uc_milp


class ScheduleValuer:
    """Coalition values of a FIXED schedule: units in S produce their scheduled output."""

    def __init__(self, grid, cfg, pg_enc, on_enc, price, sig, dt, netload):
        self.grid, self.cfg = grid, cfg
        self.pg, self.on = pg_enc, on_enc
        self.price, self.sig, self.dt = price, sig, dt
        U = grid.units
        enc = grid.encoded
        self.a, self.b, self.c = (U[k].to_numpy()[enc] for k in ("a", "b", "c"))
        self.pmax = U.pmax.to_numpy()[enc]
        self.netload = netload

    def __call__(self, masks):
        bits = masks_to_bits(masks, self.grid.n_enc)
        p = bits * self.pg
        on = bits * self.on
        o, dt = self.cfg["objectives"], self.dt
        energy = p.sum(1) * dt
        R = self.price * energy
        Cg = (self.a * p ** 2 + self.b * p + self.c * on).sum(1) * dt
        Cn = o["network_cost_per_mwh"] * energy
        kappa = (bits * self.pmax).sum(1) / self.pmax.sum()
        Cm = o["imbalance_price_factor"] * abs(self.price) * np.clip(kappa * self.netload - p.sum(1), 0, None) * dt
        m = R - Cg - Cn - Cm
        w = self.cfg["stage4"]["spread_weights"]
        s = np.sqrt(w["res"] * (abs(self.price) * self.sig[0] * kappa * dt) ** 2
                    + w["load"] * (abs(self.price) * self.sig[1] * kappa * dt) ** 2
                    + w["price"] * (self.sig[2] * energy) ** 2)
        empty = np.asarray(masks) == 0
        m[empty] = 0
        s[empty] = 0
        return m, s


class DayAhead:
    def __init__(self, grid, cfg, variant="FBWM", rng=None):
        assert variant in ("FBWM", "GFN-SV")
        self.grid, self.cfg, self.variant = grid, cfg, variant
        self.label = variant
        self.rng = rng or np.random.default_rng(0)
        self.dt = cfg["data"]["interval_minutes"] / 60

    def _weights(self, sc):
        if self.variant == "GFN-SV":
            return np.ones(3) / 3
        cfg = self.cfg
        s = np.mean([context_vector(cfg, sc.ppv_da[t], sc.ppv_da[t], sc.price_da[t], sc.price_ref,
                                    cfg["stage1"]["freq_nominal"]) for t in range(sc.T)], axis=0)
        return FBWM(cfg, adaptive=True)(s)

    def run_day(self, sc, intervals=None):
        grid, cfg = self.grid, self.cfg
        T = sc.T
        t0 = time.perf_counter()
        w = self._weights(sc)
        D = sc.pd_da.sum(1)[None, :]
        R = sc.ppv_da.sum(1)[None, :]
        u_all, p_all, status, _ = uc_milp(grid, cfg, D, R, np.array([1.0]), w,
                                          time_limit=cfg["baselines"]["day_ahead_milp_time_limit_s"],
                                          co2=sc.co2_t_per_mwh, rho=grid.security_sensitivity(None))
        t_uc = (time.perf_counter() - t0) / T
        if u_all is None:
            u_all = np.ones((T, grid.n_units))
            p_all = None
        out, u_prev = [], None
        sig = (forecast_error_sd(sc.ppv.sum(1), sc.ppv_da.sum(1)),
               forecast_error_sd(sc.pd.sum(1), sc.pd_da.sum(1)),
               forecast_error_sd(sc.price_rt, sc.price_da))
        for t in (intervals if intervals is not None else range(T)):
            u = u_all[t]
            t1 = time.perf_counter()
            r = solve_acopf(grid, sc.pd_da[t], sc.qd_da[t], sc.ppv_da[t], u)
            t_opf = time.perf_counter() - t1
            ok = r.success
            extra = {"status": status, "w_cost": w[0], "w_carbon": w[1], "w_grid": w[2], "acopf_ok": bool(ok)}
            if ok:
                pg, vg, planned = r.pg, r.vg, (r.vm, r.va, r.pg, r.qg)
            elif p_all is not None:
                # AC-OPF on the forecast failed: the raw output is the DC dispatch of the UC
                pg, vg = p_all[0, t], default_vg(grid)
                planned = planned_state_pf(grid, sc.pd_da[t], sc.qd_da[t], sc.ppv_da[t], u, pg, vg)
            else:
                pg = vg = planned = None
            t2 = time.perf_counter()
            if pg is not None:
                extra.update(self._settle(sc, t, u, pg, sig))
            t_set = time.perf_counter() - t2
            rec = record_issue(self.label, grid, sc, t, u, pg, vg, u_prev,
                               {"uc_amortized": t_uc, "acopf": t_opf, "set": t_set},
                               planned=planned, extra=extra)
            out.append(rec)
            u_prev = u
        return out

    def _settle(self, sc, t, u, pg, sig):
        grid, cfg = self.grid, self.cfg
        enc = grid.encoded
        fuzzy = self.variant == "GFN-SV"
        s = sig if fuzzy else (0.0, 0.0, 0.0)
        nl = max(sc.pd_da[t].sum() - sc.ppv_da[t].sum(), 0.0)
        val = ScheduleValuer(grid, cfg, pg[enc], u[enc], sc.price_da[t], s, self.dt, nl)
        s4 = cfg["stage4"]
        n = grid.n_enc
        m_phi, s_phi, info = shapley(n, val, s4["shapley_exact_max_n"], s4["n_permutations"], self.rng)
        m_grand, _ = val(np.array([(1 << n) - 1]))
        res = {"e_coal": coalition_efficiency_error(m_phi, float(m_grand[0])),
               "sigma_rel_mean": float(np.mean(relative_spread(m_phi, s_phi)))}
        if fuzzy and s4.get("eval_coverage", True):
            real = ScheduleValuer(grid, cfg, pg[enc], u[enc], sc.price_rt[t], (0, 0, 0), self.dt,
                                  max(sc.pd[t].sum() - sc.ppv[t].sum(), 0))
            m_real, _, _ = shapley(n, real, s4["shapley_exact_max_n"], s4["n_permutations"],
                                   np.random.default_rng(t))
            res.update(alloc_m=m_phi.tolist(), alloc_s=s_phi.tolist(), alloc_real=m_real.tolist())
        return res
