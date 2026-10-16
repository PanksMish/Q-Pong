"""Offline training data: reference AC-OPF solutions on training/validation days.

For each sample: a random (day, interval) of the split, a random commitment
of the encoded units that passes the adequacy check (fixed units on), and
the realized load/PV of that interval; the AC-OPF solution is the target.
Infeasible OPFs are skipped. Generation is parallel over processes.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..grid.acopf import capacity_ok, solve_acopf
from .model import feature_vector, target_vector

_G = {}


def _init(cfg):
    from ..data import load_profiles
    from ..grid import GridModel

    _G["grid"] = GridModel.from_config(cfg)
    _G["prof"], _G["meta"] = load_profiles(cfg)
    _G["cfg"] = cfg
    _G["scen"] = {}


def _scenario(day, pen):
    from ..data import make_scenario

    key = (day, pen)
    if key not in _G["scen"]:
        _G["scen"][key] = make_scenario(_G["grid"], _G["prof"], _G["meta"], _G["cfg"], day, pen)
    return _G["scen"][key]


def _sample(args):
    day, t, pen, seed, all_on = args
    grid, cfg = _G["grid"], _G["cfg"]
    rng = np.random.default_rng(seed)
    sc = _scenario(day, pen)
    pd, qd, ppv = sc.pd[t], sc.qd[t], sc.ppv[t]
    net = pd.sum() - ppv.sum()
    for _ in range(20):
        u_enc = np.ones(grid.n_enc) if all_on else (rng.random(grid.n_enc) < rng.uniform(0.5, 1.0)).astype(float)
        u = grid.full_commitment(u_enc)
        if capacity_ok(grid, u, net, cfg["grid"]["reserve_fraction"]):
            break
    else:
        return None
    r = solve_acopf(grid, pd, qd, ppv, u)
    if not r.success:
        return None
    x = feature_vector(grid, u, pd, qd, ppv, sc.price_rt[t], sc.price_ref)
    y = target_vector(grid, r.pg, r.qg, r.vm, r.va)
    return x, y


def generate(cfg, days, n_samples, seed=0, n_jobs=1, all_on=False):
    rng = np.random.default_rng(seed)
    pens = cfg["data"]["penetrations"]
    T = cfg["data"]["intervals_per_day"]
    tasks = [(int(rng.choice(days)), int(rng.integers(T)), float(rng.choice(pens)),
              int(rng.integers(2 ** 31)), all_on) for _ in range(n_samples)]
    if n_jobs == 1:
        _init(cfg)
        out = [_sample(a) for a in tasks]
    else:
        with ProcessPoolExecutor(n_jobs, initializer=_init, initargs=(cfg,)) as ex:
            out = list(ex.map(_sample, tasks, chunksize=16))
    out = [o for o in out if o is not None]
    X = np.stack([o[0] for o in out])
    Y = np.stack([o[1] for o in out])
    return X, Y, len(tasks) - len(out)
