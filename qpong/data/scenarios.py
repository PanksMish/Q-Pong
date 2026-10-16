"""Scenario construction: map measured profiles onto an IEEE case (eq. 24).

    D_t      = D0 * L_t / median(L)                (bus loads scaled proportionally)
    P_RES_t  = P_RES_max * R_t / max_tau R_tau      (PV at the largest load buses)

P_RES_max is set so that PV supplies ``penetration`` of the annual energy
demand (MODELING CHOICE: the paper gives the penetration levels, not their
definition). Stress classes (+15% demand, -30% renewables, both) are applied
after normalization and identically for every method.

Forecasts (MODELING CHOICE, documented in configs/default.yaml):
  * intraday   : persistence, x_hat(t) = x(t-1)
  * day-ahead  : same interval of the previous day, x_hat(t) = x(t-96)

Frequency is not contained in any of the four data sources. It is derived
from the net-load forecast error through an aggregate frequency response
(MODELING CHOICE): Delta f_t = -(netload_t - netload_hat_t) / (beta * D_t),
with beta = ``freq_response_per_hz`` (fraction of load per Hz).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    system: str
    day: int
    penetration: float
    stress: str
    pd: np.ndarray = field(repr=False)        # (T, nb) realized active load, MW
    qd: np.ndarray = field(repr=False)        # (T, nb)
    ppv: np.ndarray = field(repr=False)       # (T, nb) realized PV, MW
    pd_id: np.ndarray = field(repr=False)     # intraday forecasts
    ppv_id: np.ndarray = field(repr=False)
    pd_da: np.ndarray = field(repr=False)     # day-ahead forecasts
    qd_da: np.ndarray = field(repr=False)
    ppv_da: np.ndarray = field(repr=False)
    price_rt: np.ndarray = field(repr=False)  # $/MWh
    price_da: np.ndarray = field(repr=False)
    price_ref: float = 0.0
    freq: np.ndarray = field(repr=False, default=None)
    co2_t_per_mwh: float = 0.4
    data_source: str = "real"

    @property
    def T(self) -> int:
        return self.pd.shape[0]

    def state(self, t: int) -> dict:
        return {k: getattr(self, k)[t] for k in ("pd", "qd", "ppv", "pd_id", "ppv_id", "pd_da",
                                                  "qd_da", "ppv_da", "price_rt", "price_da", "freq")}


def pv_buses(base_p: np.ndarray, frac: float) -> np.ndarray:
    order = np.argsort(-base_p, kind="stable")
    k = max(1, int(round(frac * np.count_nonzero(base_p > 0))))
    return np.sort(order[:k])


def make_scenario(grid, profiles: pd.DataFrame, meta: dict, cfg: dict, day: int,
                  penetration: float = 0.2, stress: str = "normal") -> Scenario:
    d = cfg["data"]
    T = d["intervals_per_day"]
    L_year = profiles["load_kw"].to_numpy()
    R_year = profiles["pv_avail"].to_numpy()
    med_L = np.median(L_year)
    D0 = grid.base_p.sum()
    w_p = grid.base_p / grid.base_p.sum()
    pf = d["load_power_factor"]
    # base Q share of each bus; buses without base Q get constant power factor
    q_ratio = np.divide(grid.base_q, grid.base_p, out=np.full_like(grid.base_p, np.tan(np.arccos(pf))),
                        where=grid.base_p > 0)

    norm_mode = d.get("demand_normalization", "median")
    if norm_mode == "median":
        D_year = D0 * L_year / med_L
    elif norm_mode == "peak":
        D_year = d.get("peak_load_factor", 1.0) * D0 * L_year / L_year.max()
    else:
        raise ValueError(norm_mode)
    R_norm = R_year / max(R_year.max(), 1e-12)
    pres_max = penetration * D_year.sum() / max(R_norm.sum(), 1e-12)
    P_year = pres_max * R_norm

    def window(arr, offset):
        start = day * T - offset
        idx = np.clip(np.arange(start, start + T), 0, len(arr) - 1)
        return arr[idx]

    D, Dp, Dd = window(D_year, 0), window(D_year, 1), window(D_year, T)
    P, Pp, Pd = window(P_year, 0), window(P_year, 1), window(P_year, T)
    if stress in ("high_demand", "combined"):
        D, Dp, Dd = (x * d["stress"]["high_demand"] for x in (D, Dp, Dd))
    if stress in ("low_renewables", "combined"):
        P, Pp, Pd = (x * d["stress"]["low_renewables"] for x in (P, Pp, Pd))

    # PV curtailment (MODELING CHOICE, identical for all methods): keep the net load
    # at or above ``min_netload_fraction`` of demand; the excess PV is curtailed.
    f_min = d.get("min_netload_fraction", 0.2)
    P, Pp, Pd = (np.minimum(p_, (1 - f_min) * dd) for p_, dd in ((P, D), (Pp, Dp), (Pd, Dd)))

    pvb = pv_buses(grid.base_p, d["pv_buses_fraction"])
    w_pv = np.zeros(grid.n_bus)
    w_pv[pvb] = grid.base_p[pvb] / grid.base_p[pvb].sum()

    def to_bus(total, w):
        return np.outer(total, w)

    pd_ = to_bus(D, w_p)
    pr = profiles
    price_rt = window(pr["price_rt"].to_numpy(), 0)
    price_da = window(pr["price_da"].to_numpy(), 0)
    net = D - P
    net_hat = Dp - Pp
    beta = cfg["stage1"].get("freq_response_per_hz", 1.0)
    f_nom = cfg["stage1"]["freq_nominal"]
    freq = f_nom - (net - net_hat) / (beta * np.maximum(D, 1e-6))

    return Scenario(
        system=grid.name, day=day, penetration=penetration, stress=stress,
        pd=pd_, qd=pd_ * q_ratio, ppv=to_bus(P, w_pv),
        pd_id=to_bus(Dp, w_p), ppv_id=to_bus(Pp, w_pv),
        pd_da=to_bus(Dd, w_p), qd_da=to_bus(Dd, w_p) * q_ratio, ppv_da=to_bus(Pd, w_pv),
        price_rt=price_rt, price_da=price_da, price_ref=float(np.median(pr["price_rt"])),
        freq=freq, co2_t_per_mwh=float(meta.get("co2_t_per_mwh", 0.4)),
        data_source=meta.get("data_source", "real"),
    )
