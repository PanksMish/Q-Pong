"""MOST baseline: MATPOWER Optimal Scheduling Tool (Murillo-Sanchez et al., 2013).

This is a Python re-implementation of MOST's formulation as
used here: a rolling-horizon, two-stage stochastic DC unit commitment over
``horizon_intervals`` with ``n_scenarios`` renewable scenarios (forecast and
+/- 1 standard deviation of the intraday forecast error, probabilities
0.25/0.5/0.25), start-up costs, ramping, min up/down times, spinning reserve
and DC line limits. The first interval's commitment and central-scenario
dispatch are issued (P set-points; case voltage set-points). It is NOT the
MATPOWER MOST code itself (MOST runs in MATLAB/Octave); if the paper's MOST
numbers came from MATPOWER MOST, either rerun them there or describe this
baseline as a MOST-style stochastic UC.
"""
from __future__ import annotations

import time

import numpy as np

from .common import default_vg, planned_state_pf, record_issue, uc_milp


class MOST:
    label = "MOST"

    def __init__(self, grid, cfg, rng=None):
        self.grid, self.cfg = grid, cfg
        self.b = cfg["baselines"]["most"]
        self.dt = cfg["data"]["interval_minutes"] / 60

    def _scenarios(self, sc, t, H):
        T = sc.T
        idx = np.clip(np.arange(t, t + H), 0, T - 1)
        d = sc.pd_id[idx].sum(1)
        r = sc.ppv_id[idx].sum(1)
        lo = max(0, t - 96)
        err = sc.ppv[lo:t + 1].sum(1) - sc.ppv_id[lo:t + 1].sum(1)
        sd = float(np.std(err)) if len(err) > 2 else 0.1 * float(r.mean() + 1e-6)
        ns = self.b["n_scenarios"]
        if ns == 1:
            return d[None], r[None], np.array([1.0])
        z = np.array([-1.0, 0.0, 1.0])
        pv = np.clip(r[None] + z[:, None] * sd, 0, None)
        return np.repeat(d[None], 3, 0), pv, np.array([0.25, 0.5, 0.25])

    def run_day(self, sc, intervals=None):
        grid, cfg = self.grid, self.cfg
        out, u_prev = [], None
        H = self.b["horizon_intervals"]
        w = np.array([1.0, 0.0, 0.0])          # MOST minimizes expected cost
        for t in (intervals if intervals is not None else range(sc.T)):
            t0 = time.perf_counter()
            D, R, pr = self._scenarios(sc, t, H)
            u, p, status, _ = uc_milp(grid, cfg, D, R, pr, w, u0=u_prev, time_limit=cfg["baselines"]["day_ahead_milp_time_limit_s"],
                                      co2=sc.co2_t_per_mwh)
            el = time.perf_counter() - t0
            if u is None:
                out.append(record_issue(self.label, grid, sc, t, np.ones(grid.n_units), None, None, u_prev,
                                        {"solve": el}, extra={"status": status}))
                continue
            ut = u[0]
            mid = len(pr) // 2
            pg = p[mid, 0]
            vg = default_vg(grid)
            plan = planned_state_pf(grid, sc.pd_id[t], sc.qd[t], sc.ppv_id[t], ut, pg, vg)
            out.append(record_issue(self.label, grid, sc, t, ut, pg, vg, u_prev, {"solve": el},
                                    planned=plan, extra={"status": status}))
            u_prev = ut
        return out
