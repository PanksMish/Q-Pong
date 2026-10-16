"""PINN-OPF baseline (Nellikkath & Chatzivasileiadis, 2022).

A physics-informed network that maps the operating condition to the AC-OPF
solution for a fixed network, used here as in its source without unit
commitment: every unit is committed. It is trained on AC-OPF solutions with
all units on (``scripts/train_pinn.py --variant pinn_opf``), with the data
term plus the AC-residual and limit penalties of eqs. (36)-(39). (The
original paper additionally penalizes KKT-condition residuals; that term is
not implemented here.) Its raw prediction is issued without verification.
"""
from __future__ import annotations

import time

import numpy as np

from ..stage3_pinn.model import feature_vector
from ..stage3_pinn.train import predict
from ..stage3_pinn.verify import setpoints_from_prediction, split_prediction
from .common import record_issue


class PINNOPF:
    label = "PINN-OPF"

    def __init__(self, grid, cfg, model, rng=None):
        self.grid, self.cfg, self.model = grid, cfg, model

    def run_day(self, sc, intervals=None):
        grid = self.grid
        out, u_prev = [], None
        u = np.ones(grid.n_units)
        for t in (intervals if intervals is not None else range(sc.T)):
            t0 = time.perf_counter()
            x = feature_vector(grid, u, sc.pd[t], sc.qd[t], sc.ppv[t], sc.price_rt[t], sc.price_ref)
            y = predict(self.model, x)[0]
            pg, vg = setpoints_from_prediction(grid, y, u)
            el = time.perf_counter() - t0
            ppg, pqg, vm, va = split_prediction(grid, y)
            out.append(record_issue(self.label, grid, sc, t, u, pg, vg, u_prev, {"predict": el},
                                    planned=(vm, va, ppg, pqg)))
            u_prev = u
        return out
