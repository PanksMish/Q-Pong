"""AC power flow, AC mismatch and violation counting (eqs. 17, 27, 34-35).

``evaluate_dispatch`` is the single independent checker used both for the
Q-PONG verification gate (Stage III) and for the physical-feasibility metrics
of every method (Section V-E), so all methods are judged by the same AC test.

Issued set-points (P of non-slack committed units, |V| of committed units)
are applied under the realized load and PV of the interval and solved with
MATPOWER's Newton-Raphson power flow (PYPOWER ``runpf``); the slack unit
absorbs the residual. The solution is checked against voltage (N_V),
thermal (N_S), generator P/Q (N_G) limits and bus power balance (N_bal).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pypower import api as ppapi
from pypower.idx_bus import VA, VM
from pypower.idx_gen import PG, QG

from .network import GridModel

_PF_OPT = ppapi.ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-8, PF_MAX_IT=30, ENFORCE_Q_LIMS=0)


@dataclass
class DispatchResult:
    converged: bool
    feasible: bool
    n_v: int = 0
    n_s: int = 0
    n_g: int = 0
    n_bal: int = 0
    pg: np.ndarray = field(default=None, repr=False)     # MW per unit, as solved
    qg: np.ndarray = field(default=None, repr=False)
    vm: np.ndarray = field(default=None, repr=False)
    va: np.ndarray = field(default=None, repr=False)     # rad
    max_vdev: float = 0.0
    max_loading: float = 0.0

    @property
    def n_viol(self) -> int:
        return self.n_v + self.n_s + self.n_g + self.n_bal


def bus_injections_pu(grid: GridModel, pg, qg, pd_mw, qd_mvar, ppv_mw):
    C = grid.unit_bus_matrix()
    p = (C @ np.nan_to_num(pg) - pd_mw + ppv_mw) / grid.base_mva
    q = (C @ np.nan_to_num(qg) - qd_mvar) / grid.base_mva
    return p, q


def calc_injections_pu(grid: GridModel, vm, va):
    """Eq. (34)."""
    th = va[:, None] - va[None, :]
    P = vm * ((grid.G * np.cos(th) + grid.B * np.sin(th)) @ vm)
    Q = vm * ((grid.G * np.sin(th) - grid.B * np.cos(th)) @ vm)
    return P, Q


def balance_residuals(grid, vm, va, pg, qg, pd_mw, qd_mvar, ppv_mw):
    pinj, qinj = bus_injections_pu(grid, pg, qg, pd_mw, qd_mvar, ppv_mw)
    pc, qc = calc_injections_pu(grid, vm, va)
    return pinj - pc, qinj - qc


def ac_mismatch(grid, vm, va, pg, qg, pd_mw, qd_mvar, ppv_mw) -> float:
    """E_AC of eq. (27): mean |r_P| + |r_Q| (pu) of a state against the realized injections."""
    rp, rq = balance_residuals(grid, vm, va, pg, qg, pd_mw, qd_mvar, ppv_mw)
    return float(np.mean(np.abs(rp) + np.abs(rq)))


def branch_loading(grid: GridModel, vm, va) -> np.ndarray:
    V = vm * np.exp(1j * va)
    sf = np.abs(V[grid.f_bus] * np.conj(grid.Yf @ V))
    st = np.abs(V[grid.t_bus] * np.conj(grid.Yt @ V))
    s = np.maximum(sf, st)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(grid.s_max), s / grid.s_max, 0.0)


def count_violations(grid: GridModel, vm, va, pg, qg, u, pd_mw, qd_mvar, ppv_mw, tol_pu=1e-4):
    vs = grid.cfg["grid"].get("vm_slack", 1e-4)
    n_v = int(np.sum((vm < grid.vmin - vs) | (vm > grid.vmax + vs)))
    load = branch_loading(grid, vm, va)
    # absolute tolerance tol_pu on the apparent-power limit (same tolerance as the balance check)
    with np.errstate(invalid="ignore"):
        excess = np.where(np.isfinite(grid.s_max), (load - 1.0) * grid.s_max, 0.0)
    n_s = int(np.sum(excess > tol_pu))
    U = grid.units
    tol_mw = tol_pu * grid.base_mva
    on = u > 0.5
    at_ref = (U.bus == grid.slack_bus).to_numpy()
    p_lo = np.where(on, U.pmin, 0.0) - tol_mw
    p_hi = np.where(on, U.pmax, 0.0) + tol_mw
    q_on = on | at_ref
    bad_p = (pg < p_lo) | (pg > p_hi)
    bad_q = q_on & ((qg < U.qmin - tol_mw) | (qg > U.qmax + tol_mw))
    n_g = int(np.sum(bad_p | bad_q))
    rp, rq = balance_residuals(grid, vm, va, pg, qg, pd_mw, qd_mvar, ppv_mw)
    n_bal = int(np.sum((np.abs(rp) > tol_pu) | (np.abs(rq) > tol_pu)))
    vdev = float(np.max(np.maximum(grid.vmin - vm, vm - grid.vmax).clip(min=0)))
    return n_v, n_s, n_g, n_bal, vdev, load


def run_pf(grid: GridModel, pd_mw, qd_mvar, ppv_mw, u, pg_mw, vg_pu):
    ppc = grid.configured_case(pd_mw, qd_mvar, ppv_mw, u, pg_mw=pg_mw, vg_pu=vg_pu)
    try:
        res, ok = ppapi.runpf(ppc, _PF_OPT)
    except Exception:
        return None
    if not ok:
        return None
    vm = res["bus"][:, VM]
    va = np.deg2rad(res["bus"][:, VA])
    pg = res["gen"][:, PG].copy()
    qg = res["gen"][:, QG].copy()
    off = (u < 0.5) & (grid.units.bus != grid.slack_bus).to_numpy()
    pg[off] = 0.0
    qg[off] = 0.0
    return vm, va, pg, qg


def evaluate_dispatch(grid: GridModel, pd_mw, qd_mvar, ppv_mw, u, pg_mw, vg_pu) -> DispatchResult:
    """Apply set-points under (realized) load/PV, run the AC PF, count violations."""
    tol = grid.cfg["grid"]["verification_tol_pu"]
    out = run_pf(grid, pd_mw, qd_mvar, ppv_mw, u, pg_mw, vg_pu)
    if out is None:
        return DispatchResult(converged=False, feasible=False, n_bal=grid.n_bus)
    vm, va, pg, qg = out
    n_v, n_s, n_g, n_bal, vdev, load = count_violations(grid, vm, va, pg, qg, u, pd_mw, qd_mvar,
                                                         ppv_mw, tol)
    feas = (n_v + n_s + n_g + n_bal) == 0
    return DispatchResult(True, feas, n_v, n_s, n_g, n_bal, pg, qg, vm, va, vdev,
                          float(load.max()) if len(load) else 0.0)
