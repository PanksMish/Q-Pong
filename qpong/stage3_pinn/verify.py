"""Independent verification gate (eqs. 16-17), local repair and classical fallback (eq. 18).

The gate admits candidate k only if
  * V_phys^(k) <= tau_phys, where V_phys is computed on the PINN prediction
    (eq. 16: residual norms of eq. 34 plus limit-violation counts), and
  * the issued set-points (P of committed units, |V| at unit buses), solved
    by an AC power flow under the realized load/PV, show
    N_V = N_S = N_G = N_bal = 0 (eq. 17).
Together with the fallback this guarantees that every issued dispatch is
feasible, up to failures of the fallback solver itself (reported separately).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid.acopf import solve_acopf
from ..grid.acpf import balance_residuals, branch_loading, evaluate_dispatch


@dataclass
class GateResult:
    admitted: bool
    v_phys: float
    dispatch: object        # DispatchResult of the issued set-points
    pg: np.ndarray
    vg: np.ndarray
    repaired: bool = False


def split_prediction(grid, y):
    ng, nb, bm = grid.n_units, grid.n_bus, grid.base_mva
    pg = y[:ng] * bm
    qg = y[ng:2 * ng] * bm
    vm = y[2 * ng:2 * ng + nb]
    va = y[2 * ng + nb:]
    return pg, qg, vm, va


def v_phys(grid, cfg, y, u, pd, qd, ppv):
    """Eq. (16) on the predicted state."""
    eta = cfg["stage3"]["eta"]
    pg, qg, vm, va = split_prediction(grid, y)
    rp, rq = balance_residuals(grid, vm, va, pg, qg, pd, qd, ppv)
    n_v = int(np.sum((vm < grid.vmin) | (vm > grid.vmax)))
    n_s = int(np.sum(branch_loading(grid, vm, va) > 1.0))
    U = grid.units
    on = u > 0.5
    n_g = int(np.sum(on & ((pg < U.pmin - 1e-6) | (pg > U.pmax + 1e-6))))
    return (eta["P"] * np.linalg.norm(rp) + eta["Q"] * np.linalg.norm(rq)
            + eta["V"] * n_v + eta["S"] * n_s + eta["G"] * n_g)


def setpoints_from_prediction(grid, y, u):
    pg, qg, vm, va = split_prediction(grid, y)
    vg = vm[grid.units.bus.to_numpy()]
    return pg * (u > 0.5), vg


def local_repair(grid, pg, vg, u, pd, ppv, loss_frac=0.02):
    """Projection onto a feasible neighbourhood: clip P to [Pmin, Pmax] of committed
    units, rescale non-slack output toward the net load plus estimated losses,
    and clip unit voltage set-points to the bus voltage limits."""
    U = grid.units
    on = u > 0.5
    pg = np.clip(pg, U.pmin * on, U.pmax * on)
    target = (pd.sum() - ppv.sum()) * (1 + loss_frac)
    nonslack = on & (U.bus != grid.slack_bus).to_numpy()
    slack_p = pg[~nonslack].sum()
    s = pg[nonslack].sum()
    if s > 0:
        pg[nonslack] = np.clip(pg[nonslack] * max(target - slack_p, 0) / s, U.pmin[nonslack], U.pmax[nonslack])
    b = U.bus.to_numpy()
    vg = np.clip(vg, grid.vmin[b] + 0.005, grid.vmax[b] - 0.005)
    return pg, vg


def gate(grid, cfg, y, u, pd, qd, ppv, verification=True) -> GateResult:
    vp = v_phys(grid, cfg, y, u, pd, qd, ppv)
    pg, vg = setpoints_from_prediction(grid, y, u)
    disp = evaluate_dispatch(grid, pd, qd, ppv, u, pg, vg)
    if not verification:
        return GateResult(True, vp, disp, pg, vg)
    ok = vp <= cfg["stage3"]["tau_phys"] and disp.feasible
    if ok:
        return GateResult(True, vp, disp, pg, vg)
    for _ in range(cfg["stage3"]["max_repair_attempts"]):
        pg2, vg2 = local_repair(grid, pg, vg, u, pd, ppv)
        d2 = evaluate_dispatch(grid, pd, qd, ppv, u, pg2, vg2)
        if d2.feasible:
            return GateResult(True, vp, d2, pg2, vg2, repaired=True)
    return GateResult(False, vp, disp, pg, vg)


def classical_fallback(grid, pd, qd, ppv, u_candidates, max_tries=10):
    """S_fallback (eq. 18): AC-OPF on the ranked candidate commitments (up to ``max_tries``),
    then a merit-order (priority-list) commitment for the net load, then all units on."""
    from ..baselines.common import priority_list

    net = float(pd.sum() - ppv.sum())
    tried = list(u_candidates)[:max_tries] + [priority_list(grid, net, grid.cfg["grid"]["reserve_fraction"]),
                                              np.ones(grid.n_units)]
    first_ok = None
    for u in tried:
        r = solve_acopf(grid, pd, qd, ppv, u)
        if r.success:
            d = evaluate_dispatch(grid, pd, qd, ppv, u, r.pg, r.vg)
            if d.feasible:
                return u, r.pg, r.vg, d, r
            first_ok = first_ok or (u, r.pg, r.vg, d, r)
    return first_ok or (tried[-1], None, None, None, None)
