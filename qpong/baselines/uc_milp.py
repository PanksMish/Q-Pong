"""Two-stage stochastic DC unit-commitment MILP (vectorized assembly, HiGHS via scipy).

Used by the day-ahead baselines (FBWM, GFN-SV; one scenario, T = 96) and by the
MOST-style rolling stochastic UC (three scenarios, T = horizon).

Variables
  u[t,i]  commitment (binary for encoded units, fixed to 1 for the others)
  v[t,i]  start-up indicator (encoded units)
  g[s,t,i,k]  output above Pmin in segment k (piecewise-linear quadratic cost)
  p[s,t,i]    total output  = Pmin u + sum_k g
  c[s,t]      load-shedding slack (penalized at VOLL)
  r[t]        spinning-reserve shortfall slack (penalized at VOLL)
Constraints
  power balance, spinning reserve, Pmin/Pmax, ramping (relaxed by Pmax at
  start-up / shut-down), min up/down time (Rajan-Takriti), DC line limits via PTDFs.
Objective (normalized as eq. (1) by the all-committed nominal values)
  w_cost f_cost + w_carbon f_carbon + w_grid f_grid  (+ slack penalties)
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..data.scenarios import pv_buses


class _Rows:
    def __init__(self):
        self.r, self.c, self.v, self.lo, self.hi = [], [], [], [], []
        self.n = 0

    def add(self, rows, cols, vals, lo, hi):
        """rows are local (0..m-1); lo/hi have length m."""
        m = len(lo)
        self.r.append(np.asarray(rows) + self.n)
        self.c.append(np.asarray(cols))
        self.v.append(np.asarray(vals, float))
        self.lo.append(np.asarray(lo, float))
        self.hi.append(np.asarray(hi, float))
        self.n += m

    def build(self, N):
        A = coo_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.c))),
                       shape=(self.n, N)).tocsr()
        return A, np.concatenate(self.lo), np.concatenate(self.hi)


def uc_milp(grid, cfg, demand, pv, probs, weights, u0=None, up0=None, down0=None, time_limit=300,
            network=True, co2=0.4, rho=None, n_seg=3):
    g = cfg["grid"]
    demand, pv = np.atleast_2d(demand).astype(float), np.atleast_2d(pv).astype(float)
    S, T = demand.shape
    U = grid.units
    ng = grid.n_units
    enc = np.asarray(grid.encoded)
    is_enc = np.zeros(ng, bool)
    is_enc[enc] = True
    pmax, pmin = U.pmax.to_numpy(), U.pmin.to_numpy()
    a, b, c = U.a.to_numpy(), U.b.to_numpy(), U.c.to_numpy()
    su, ramp = U.startup.to_numpy(), U.ramp.to_numpy()
    dt = cfg["data"]["interval_minutes"] / 60
    rho = np.zeros(ng) if rho is None else np.asarray(rho, float)
    probs = np.asarray(probs, float)
    w_cost, w_carb, w_grid = weights

    # ---- variable index blocks
    nU = T * ng
    _off_u, off_v = 0, nU
    off_g = 2 * nU
    nG = S * T * ng * n_seg
    off_p = off_g + nG
    nP = S * T * ng
    off_c = off_p + nP
    off_r = off_c + S * T
    N = off_r + T
    iu = (np.arange(T)[:, None] * ng + np.arange(ng)[None, :])                         # (T, ng)
    iv = off_v + iu
    ig = off_g + np.arange(nG).reshape(S, T, ng, n_seg)
    ip = off_p + np.arange(nP).reshape(S, T, ng)
    ic = off_c + np.arange(S * T).reshape(S, T)
    ir = off_r + np.arange(T)

    # ---- objective
    seg_w = (pmax - pmin) / n_seg
    seg_mc = a[:, None] * (2 * pmin[:, None] + (2 * np.arange(n_seg)[None, :] + 1) * seg_w[:, None]) + b[:, None]
    p_nom = np.clip(g["target_loading"] * pmax, pmin, pmax)
    n_cost = np.sum(a * p_nom ** 2 + b * p_nom + c) * dt + 1e-9
    n_carb = np.sum(co2 * p_nom) * dt + 1e-9
    n_grid = np.sum(rho * p_nom) + 1e-9
    voll = 100.0 * b.max() * dt / n_cost

    cvec = np.zeros(N)
    base = w_cost * (a * pmin ** 2 + b * pmin + c) * dt / n_cost + w_carb * co2 * pmin * dt / n_carb \
        + w_grid * rho * pmin / n_grid
    cvec[iu] = base[None, :]
    cvec[iv] = (w_cost * su / n_cost * is_enc)[None, :]
    seg_cost = w_cost * seg_mc * dt / n_cost + (w_carb * co2 * dt / n_carb + w_grid * rho / n_grid)[:, None]
    cvec[ig] = probs[:, None, None, None] * seg_cost[None, None, :, :]
    cvec[ic] = probs[:, None] * voll
    cvec[ir] = voll

    lb = np.zeros(N)
    ub = np.full(N, np.inf)
    ub[iu] = 1.0
    ub[iv] = 1.0
    lb[iu[:, ~is_enc]] = 1.0
    ub[iv[:, ~is_enc]] = 0.0
    ub[ig] = np.broadcast_to(seg_w[None, None, :, None], ig.shape)
    lb[ip] = 0.0
    integ = np.zeros(N)
    integ[iu[:, is_enc]] = 1

    R = _Rows()
    nl = demand - pv
    # p definition: p - pmin u - sum_k g = 0
    rows = np.arange(nP)
    s_, t_, i_ = np.unravel_index(rows, (S, T, ng))
    R.add(np.r_[rows, rows, np.repeat(rows, n_seg)],
          np.r_[ip.ravel(), iu[t_, i_], ig.reshape(nP, n_seg).ravel()],
          np.r_[np.ones(nP), -pmin[i_], -np.ones(nP * n_seg)], np.zeros(nP), np.zeros(nP))
    # p <= pmax u
    R.add(np.r_[rows, rows], np.r_[ip.ravel(), iu[t_, i_]], np.r_[np.ones(nP), -pmax[i_]],
          np.full(nP, -np.inf), np.zeros(nP))
    # balance: sum_i p + c = nl
    m = S * T
    rr = np.repeat(np.arange(m), ng)
    R.add(np.r_[rr, np.arange(m)], np.r_[ip.reshape(m, ng).ravel(), ic.ravel()],
          np.ones(m * ng + m), nl.ravel(), nl.ravel())
    # reserve: sum_i pmax u + r >= max_s nl * (1 + reserve)
    rr = np.repeat(np.arange(T), ng)
    req = nl.max(axis=0) * (1 + g["reserve_fraction"])
    R.add(np.r_[rr, np.arange(T)], np.r_[iu.ravel(), ir], np.r_[np.tile(pmax, T), np.ones(T)],
          req, np.full(T, np.inf))
    # start-up: v_t - u_t + u_{t-1} >= 0
    E = enc
    ne = len(E)
    if T > 1:
        k = (T - 1) * ne
        rr = np.arange(k)
        tt = np.repeat(np.arange(1, T), ne)
        ii = np.tile(E, T - 1)
        R.add(np.r_[rr, rr, rr], np.r_[iv[tt, ii], iu[tt, ii], iu[tt - 1, ii]],
              np.r_[np.ones(k), -np.ones(k), np.ones(k)], np.zeros(k), np.full(k, np.inf))
    if u0 is not None:
        R.add(np.r_[np.arange(ne), np.arange(ne)], np.r_[iv[0, E], iu[0, E]], np.r_[np.ones(ne), -np.ones(ne)],
              -np.asarray(u0, float)[E], np.full(ne, np.inf))
    # ramping on p, relaxed by pmax at start-up / shut-down
    if T > 1:
        k = S * (T - 1) * ng
        rr = np.arange(k)
        s_, t_, i_ = np.unravel_index(rr, (S, T - 1, ng))
        t_ = t_ + 1
        # up: p_t - p_{t-1} + pmax u_{t-1} <= ramp + pmax
        R.add(np.r_[rr, rr, rr], np.r_[ip[s_, t_, i_], ip[s_, t_ - 1, i_], iu[t_ - 1, i_]],
              np.r_[np.ones(k), -np.ones(k), pmax[i_]], np.full(k, -np.inf), ramp[i_] + pmax[i_])
        # down: p_t - p_{t-1} - pmax u_t >= -ramp - pmax
        R.add(np.r_[rr, rr, rr], np.r_[ip[s_, t_, i_], ip[s_, t_ - 1, i_], iu[t_, i_]],
              np.r_[np.ones(k), -np.ones(k), -pmax[i_]], -ramp[i_] - pmax[i_], np.full(k, np.inf))
    # min up / down (Rajan-Takriti) with start-ups only
    UT, DT = g["min_up_intervals"], g["min_down_intervals"]
    rows_r, cols_r, vals_r, lo_r, hi_r = [], [], [], [], []
    r0 = 0
    for i in E:
        for t in range(T):
            win = np.arange(max(0, t - UT + 1), t + 1)
            rows_r += [r0] * (len(win) + 1)
            cols_r += list(iv[win, i]) + [iu[t, i]]
            vals_r += [1.0] * len(win) + [-1.0]
            lo_r.append(-np.inf)
            hi_r.append(0.0)
            r0 += 1
            win = np.arange(max(0, t - DT + 1), t + 1)
            if t - DT >= 0:
                rows_r += [r0] * (len(win) + 1)
                cols_r += list(iv[win, i]) + [iu[t - DT, i]]
                vals_r += [1.0] * (len(win) + 1)
                lo_r.append(-np.inf)
                hi_r.append(1.0)
                r0 += 1
            elif u0 is not None:
                rows_r += [r0] * len(win)
                cols_r += list(iv[win, i])
                vals_r += [1.0] * len(win)
                lo_r.append(-np.inf)
                hi_r.append(1.0 - float(u0[i]))
                r0 += 1
    if r0:
        R.add(rows_r, cols_r, vals_r, lo_r, hi_r)
    # DC line limits: H (C p - load + pv) in [-smax, smax]
    if network:
        H = grid.ptdf()
        rated = np.nonzero(np.isfinite(grid.s_max))[0]
        if len(rated):
            Hg = H[rated][:, U.bus.to_numpy()] / grid.base_mva           # (L, ng)
            w_load = grid.base_p / grid.base_p.sum()
            pvw = np.zeros(grid.n_bus)
            pb = pv_buses(grid.base_p, cfg["data"]["pv_buses_fraction"])
            pvw[pb] = grid.base_p[pb] / grid.base_p[pb].sum()
            hl = H[rated] @ w_load / grid.base_mva
            hp = H[rated] @ pvw / grid.base_mva
            smax = grid.s_max[rated]
            len(rated)
            nz_l, nz_i = np.nonzero(np.abs(Hg) > 1e-9)
            vals = Hg[nz_l, nz_i]
            for s in range(S):
                for t in range(T):
                    off = hl * demand[s, t] - hp * pv[s, t]
                    R.add(nz_l, ip[s, t, nz_i], vals, -smax + off, smax + off)
    A, lo_b, hi_b = R.build(N)
    t0 = time.perf_counter()
    res = milp(cvec, constraints=LinearConstraint(A, lo_b, hi_b), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-4, "disp": False})
    elapsed = time.perf_counter() - t0
    if res.x is None:
        return None, None, f"failed:{res.status}", elapsed
    x = res.x
    u = np.round(x[iu])
    p = x[ip]
    shed = float(x[ic].sum())
    short = float(x[ir].sum())
    status = "optimal" if res.status == 0 else f"status:{res.status}"
    if shed > 1e-6 or short > 1e-6:
        status += f";shed={shed:.1f}MW;reserve_short={short:.1f}MW"
    return u, p, status, elapsed
