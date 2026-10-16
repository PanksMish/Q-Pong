"""Bridge: grid model + interval state -> reduced UC problem over the encoded units."""
from __future__ import annotations

import numpy as np

from .stage2_quantum.qubo import UCProblem


def make_uc_problem(grid, cfg, weights, pd, ppv, pd_hat, co2, u_prev_enc=None, up=None, down=None,
                    p_fixed=0.0, loading_prev=None) -> UCProblem:
    U = grid.units
    enc = grid.encoded
    g = cfg["grid"]
    rho = grid.security_sensitivity(loading_prev)[enc]
    p_fixed_max = float(U.pmax.to_numpy()[grid.fixed].sum()) if len(grid.fixed) else 0.0
    return UCProblem(
        pmax=U.pmax.to_numpy()[enc], pmin=U.pmin.to_numpy()[enc],
        a=U.a.to_numpy()[enc], b=U.b.to_numpy()[enc], c=U.c.to_numpy()[enc],
        startup=U.startup.to_numpy()[enc], emission=np.full(len(enc), co2), rho=rho,
        demand=float(pd.sum()), demand_hat=float(pd_hat.sum()), res=float(ppv.sum()),
        p_fixed=float(p_fixed), p_fixed_max=p_fixed_max, weights=np.asarray(weights, float),
        u_prev=None if u_prev_enc is None else np.asarray(u_prev_enc, float),
        up_time=up, down_time=down, min_up=g["min_up_intervals"], min_down=g["min_down_intervals"],
        target_loading=g["target_loading"], reserve=g["reserve_fraction"],
        dt_h=cfg["data"]["interval_minutes"] / 60.0,
    )
