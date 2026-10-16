"""AC optimal power flow for a fixed commitment (MATPOWER MIPS via PYPOWER ``runopf``).

Used for reference states x_ref (eq. 28), PINN training data, the classical
fallback S_fallback (eq. 18) and the dispatch step of several baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pypower import api as ppapi
from pypower.idx_bus import VA, VM
from pypower.idx_gen import PG, QG

from .network import GridModel

_OPF_OPT = ppapi.ppoption(VERBOSE=0, OUT_ALL=0, OPF_FLOW_LIM=0)


@dataclass
class OPFResult:
    success: bool
    pg: np.ndarray = field(default=None, repr=False)
    qg: np.ndarray = field(default=None, repr=False)
    vm: np.ndarray = field(default=None, repr=False)
    va: np.ndarray = field(default=None, repr=False)
    vg: np.ndarray = field(default=None, repr=False)
    cost: float = np.nan


def solve_acopf(grid: GridModel, pd_mw, qd_mvar, ppv_mw, u) -> OPFResult:
    ppc = grid.configured_case(pd_mw, qd_mvar, ppv_mw, u)
    try:
        res = ppapi.runopf(ppc, _OPF_OPT)
        ok = bool(res["success"])
    except Exception:
        ok = False
    if not ok:
        return OPFResult(False)
    vm = res["bus"][:, VM]
    va = np.deg2rad(res["bus"][:, VA])
    va = va - va[grid.slack_bus]
    pg = res["gen"][:, PG].copy()
    qg = res["gen"][:, QG].copy()
    off = (u < 0.5) & (grid.units.bus != grid.slack_bus).to_numpy()
    pg[off] = 0.0
    qg[off] = 0.0
    vg = vm[grid.units.bus.to_numpy()]
    return OPFResult(True, pg, qg, vm, va, vg, float(res["f"]))


def capacity_ok(grid: GridModel, u, net_load_mw, reserve=0.0) -> bool:
    U = grid.units
    return (float(np.sum(u * U.pmax)) >= net_load_mw * (1 + reserve)
            and float(np.sum(u * U.pmin)) <= max(net_load_mw, 0.0))
