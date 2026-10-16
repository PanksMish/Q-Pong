"""Experiment runner: method factory, scenario loops, JSONL result files.

Every result line is one (method, system, day, interval, penetration, stress, seed)
record. Files: results/runs/<tag>/<system>/<method>/pen<p>_<stress>_seed<s>.jsonl
"""
from __future__ import annotations

import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import resolve_path
from .data import day_splits, load_profiles, make_scenario
from .grid import GridModel
from .utils import get_logger, set_seed

log = get_logger("qpong.exp")

METHODS = ["G-MINLP", "MOST", "QDUC", "PINN-OPF", "FBWM", "GFN-SV", "Q-PONG"]
ABLATIONS = {
    "Fixed": {"ablation.fixed_weights": True},
    "Classical (SA)": {"ablation.quantum_solver": "sa"},
    "Exact": {"ablation.quantum_solver": "exact"},
    "Data-only": {"ablation.pinn_physics": False},
    "No-Verif.": {"ablation.verification": False},
    "Deterministic": {"ablation.fuzzy_settlement": False},
    "No-HITL": {"ablation.hitl": False},
}


def model_path(cfg, variant: str, seed: int) -> Path:
    return resolve_path(cfg, f"{cfg['experiment']['results_dir']}/models/{cfg['system']['name']}_{variant}_seed{seed}.pt")


def _load_model(grid, cfg, variant, seed):
    from .stage3_pinn.train import load_model

    p = model_path(cfg, variant, seed)
    if not p.exists():
        p0 = model_path(cfg, variant, 0)
        if not p0.exists():
            raise FileNotFoundError(f"{p} missing - run scripts/train_pinn.py --system {cfg['system']['name']} "
                                    f"--variant {variant}")
        log.warning("model for seed %d missing; using seed-0 model %s", seed, p0)
        p = p0
    return load_model(grid, p, act=cfg["stage3"]["activation"])[0]


def apply_overrides(cfg, ov: dict):
    cfg = copy.deepcopy(cfg)
    for k, v in ov.items():
        node = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            node = node[kk]
        node[ks[-1]] = v
    return cfg


def build_method(name, grid, cfg, seed):
    from .baselines import GMINLP, MOST, PINNOPF, QDUC, DayAhead
    from .pipeline import QPONG

    rng = np.random.default_rng(seed)
    if name == "G-MINLP":
        return GMINLP(grid, cfg, rng)
    if name == "MOST":
        return MOST(grid, cfg, rng)
    if name == "QDUC":
        return QDUC(grid, cfg, rng)
    if name == "PINN-OPF":
        return PINNOPF(grid, cfg, _load_model(grid, cfg, "pinn_opf", seed), rng)
    if name in ("FBWM", "GFN-SV"):
        return DayAhead(grid, cfg, name, rng)
    if name == "Q-PONG":
        return QPONG(grid, cfg, _load_model(grid, cfg, "qpong", seed), rng)
    if name in ABLATIONS:
        c2 = apply_overrides(cfg, ABLATIONS[name])
        variant = "data_only" if not c2["ablation"]["pinn_physics"] else "qpong"
        return QPONG(grid, c2, _load_model(grid, c2, variant, seed), rng, label=name)
    raise ValueError(name)


def _jsonable(r):
    out = {}
    for k, v in r.items():
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out


def run_job(job):
    """job = (cfg, method, penetration, stress, seed, days, intervals, out_file)."""
    cfg, method, pen, stress, seed, days, intervals, out_file = job
    set_seed(seed)
    grid = GridModel.from_config(cfg)
    prof, meta = load_profiles(cfg)
    m = build_method(method, grid, cfg, seed)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_file.exists():                       # resume
        for line in out_file.read_text().splitlines():
            r = json.loads(line)
            done.add(r["day"])
    with open(out_file, "a") as f:
        for d in days:
            if d in done:
                continue
            sc = make_scenario(grid, prof, meta, cfg, d, pen, stress)
            recs = m.run_day(sc, intervals)
            for r in recs:
                r["seed"] = seed
                f.write(json.dumps(_jsonable(r), default=str) + "\n")
            f.flush()
            log.info("%s %s pen=%.1f %s seed=%d day=%d done", cfg["system"]["name"], method, pen, stress, seed, d)
    return str(out_file)


def plan_jobs(cfg, methods, tag, penetrations=None, stresses=("normal",), seeds=None, days=None,
              intervals=None):
    sp = day_splits(stride=cfg["data"]["test_day_stride"], n_val=cfg["data"]["n_val_days"])
    days = days if days is not None else sp["test"]
    seeds = seeds if seeds is not None else cfg["experiment"]["seeds"]
    pens = penetrations if penetrations is not None else cfg["data"]["penetrations"]
    root = resolve_path(cfg, f"{cfg['experiment']['results_dir']}/runs/{tag}/{cfg['system']['name']}")
    jobs = []
    for m in methods:
        for p in pens:
            for s in stresses:
                for seed in seeds:
                    f = root / m.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "") \
                        / f"pen{int(round(p * 100))}_{s}_seed{seed}.jsonl"
                    jobs.append((cfg, m, p, s, seed, days, intervals, str(f)))
    return jobs


def run_jobs(jobs, n_jobs=1):
    if n_jobs == 1:
        return [run_job(j) for j in jobs]
    with ProcessPoolExecutor(n_jobs) as ex:
        return list(ex.map(run_job, jobs))
