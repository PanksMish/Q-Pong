"""Scalarized objective F(u, x; w) of eq. (1) evaluated on a full dispatch."""
from __future__ import annotations

import numpy as np


def normalizers(grid, cfg, co2, rho_units):
    U = grid.units
    dt = cfg["data"]["interval_minutes"] / 60
    p_all = np.clip(cfg["grid"]["target_loading"] * U.pmax, U.pmin, U.pmax).to_numpy()
    n_cost = float(np.sum(U.a * p_all ** 2 + U.b * p_all + U.c) * dt) + 1e-9
    n_carb = float(np.sum(co2 * p_all) * dt) + 1e-9
    n_grid = float(np.sum(rho_units * p_all)) + 1e-9
    return n_cost, n_carb, n_grid


def scalarized(grid, cfg, w, u, pg, u_prev, co2, rho_units):
    dt = cfg["data"]["interval_minutes"] / 60
    n_cost, n_carb, n_grid = normalizers(grid, cfg, co2, rho_units)
    f_cost = grid.gen_cost(pg, u, dt) + grid.startup_cost(u, u_prev)
    f_carb = float(np.sum(pg) * dt * co2)
    f_grid = float(np.sum(rho_units * pg))
    return float(w[0] * f_cost / n_cost + w[1] * f_carb / n_carb + w[2] * f_grid / n_grid)
