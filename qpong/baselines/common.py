"""Shared helpers for baselines: issuance/recording, economic dispatch,
priority-list commitment and the (stochastic, multi-period) DC UC MILP."""
from __future__ import annotations


import numpy as np

from ..grid.acpf import ac_mismatch, evaluate_dispatch, run_pf


def default_vg(grid):
    from pypower.idx_gen import VG

    return grid.ppc["gen"][:, VG].copy()


def record_issue(label, grid, sc, t, u, pg, vg, u_prev, times: dict, planned=None, extra=None):
    """Issue set-points (no gate): S1 = S2 = one AC check under the realized state."""
    dt = grid.cfg["data"]["interval_minutes"] / 60
    pd, qd, ppv = sc.pd[t], sc.qd[t], sc.ppv[t]
    rec = {"method": label, "t": t, "system": sc.system, "day": sc.day, "penetration": sc.penetration,
           "stress": sc.stress, "data_source": sc.data_source, "times": times,
           "t_e2e": float(sum(times.values())), "fallback": False}
    if pg is None:
        rec.update(s1_feasible=False, s2_feasible=False, s1_nv=0, s1_ns=0, s1_ng=0, s1_nbal=grid.n_bus,
                   s2_nv=0, s2_ns=0, s2_ng=0, s2_nbal=grid.n_bus, cost=np.nan, e_ac=np.nan,
                   solver_failure=True)
        return rec
    d = evaluate_dispatch(grid, pd, qd, ppv, u, pg, vg)
    for s in ("s1", "s2"):
        rec.update({f"{s}_feasible": bool(d.feasible), f"{s}_nv": d.n_v, f"{s}_ns": d.n_s,
                    f"{s}_ng": d.n_g, f"{s}_nbal": d.n_bal})
    if d.converged:
        rec["cost"] = grid.gen_cost(d.pg, u, dt) + grid.startup_cost(u, u_prev)
        rec["carbon_t"] = float(d.pg.sum() * dt * sc.co2_t_per_mwh)
    else:
        rec["cost"] = grid.gen_cost(pg, u, dt) + grid.startup_cost(u, u_prev)
    rec["energy_mwh"] = float((pd.sum() - ppv.sum()) * dt)
    if planned is not None:
        vm, va, ppg, pqg = planned
        rec["e_ac"] = ac_mismatch(grid, vm, va, ppg, pqg, pd, qd, ppv)
    elif d.converged:
        rec["e_ac"] = ac_mismatch(grid, d.vm, d.va, d.pg, d.qg, pd, qd, ppv)
    rec["n_committed"] = int(u[grid.encoded].sum())
    rec["u_enc"] = "".join(str(int(x)) for x in u[grid.encoded])
    if extra:
        rec.update(extra)
    return rec


def planned_state_pf(grid, pd, qd, ppv, u, pg, vg):
    """The state a DC-based method expects: AC PF of its set-points on its own forecast."""
    out = run_pf(grid, pd, qd, ppv, u, pg, vg)
    return out


def economic_dispatch(grid, u, net_load, loss_frac=0.02):
    """Quadratic-cost ED by bisection on the system lambda (no network)."""
    U = grid.units
    on = u > 0.5
    a, b = U.a.to_numpy(), U.b.to_numpy()
    lo, hi = (U.pmin * on).to_numpy(), (U.pmax * on).to_numpy()
    target = net_load * (1 + loss_frac)

    def p_of(lam):
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(a > 1e-12, (lam - b) / (2 * a), np.where(lam >= b, hi, lo))
        return np.clip(p, lo, hi)

    L, H = -1e3, 1e4
    for _ in range(100):
        m = 0.5 * (L + H)
        if p_of(m).sum() < target:
            L = m
        else:
            H = m
    return p_of(H)


def priority_list(grid, net_load, reserve):
    """Merit-order commitment of encoded units (average full-load cost) to cover
    net load * (1 + reserve); fixed units always on."""
    U = grid.units
    enc = grid.encoded
    pmax = U.pmax.to_numpy()
    avg = (U.a * U.pmax ** 2 + U.b * U.pmax + U.c).to_numpy() / np.maximum(pmax, 1e-6)
    u = np.ones(grid.n_units)
    u[enc] = 0
    cap = pmax[grid.fixed].sum() if len(grid.fixed) else 0.0
    for i in enc[np.argsort(avg[enc], kind="stable")]:
        if cap >= net_load * (1 + reserve):
            break
        u[i] = 1
        cap += pmax[i]
    return u


from .uc_milp import uc_milp  # noqa: E402,F401
