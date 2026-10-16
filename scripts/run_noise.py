#!/usr/bin/env python
"""Robustness to simulated hardware noise (Section VI-D, Table VII).

Runs Q-PONG with ``stage2.mode = noise`` (Aer + depolarizing/thermal-relaxation
model) next to the noiseless run, on the same scenarios and seeds, and reports
ground-state accuracy, the shot-weighted feasible share S_feas/S_shots, the
normalized cost and its increase over the noiseless cost.

    python scripts/run_noise.py --system ieee30 ieee57 ieee118 --days 11 --n-jobs 6
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from _common import base_parser, config_for, smoke_defaults

from qpong.experiments import plan_jobs, run_jobs
from qpong.metrics import load_runs


def accuracy(df):
    e = df["vqe_energy"].where(df["vqe_energy"].notna(), df.get("vqe_energy_shots"))
    return 100 * (1 - (e - df["e_exact"]).abs() / df["e_exact"].abs())


def main():
    ap = base_parser(__doc__)
    args = smoke_defaults(ap.parse_args())
    tag = args.tag or "noise"
    jobs = []
    for system in args.system:
        cfg0 = config_for(system, args)
        pen = [cfg0["data"]["penetrations"][0]]
        for mode, extra in (("noiseless", []), ("noise", ["stage2.mode=noise", "stage2.simulator=aer"])):
            a = type("A", (), {"set": list(args.set) + extra, "smoke": args.smoke})()
            cfg = config_for(system, a)
            jobs += plan_jobs(cfg, ["Q-PONG"], f"{tag}/{mode}", pen, ["normal"], args.seeds, args.days,
                              args.intervals)
    run_jobs(jobs, args.n_jobs)
    root = Path(config_for(args.system[0], args)["_repo_root"]) / config_for(args.system[0], args)[
        "experiment"]["results_dir"] / "runs" / tag
    rows = []
    for mode in ("noiseless", "noise"):
        df = load_runs(root / mode)
        if df.empty:
            continue
        df["accuracy"] = accuracy(df)
        df["feas_share"] = 100 * df["s_feas"] / df["s_shots"]
        for system, g in df.groupby("system"):
            e = g["energy_mwh"].where(np.isfinite(g["cost"]))
            rows.append({"system": system, "mode": mode, "accuracy": g["accuracy"].mean(),
                         "feas_share": g["feas_share"].mean(),
                         "cost_per_mwh": g["cost"].sum() / e.sum(), "issued_feasible": 100 * g["s2_feasible"].mean()})
    t = pd.DataFrame(rows)
    if not t.empty:
        base = t[t["mode"] == "noiseless"].set_index("system")["cost_per_mwh"]
        t["cost_increase_pct"] = [100 * (c / base.get(s, np.nan) - 1) for s, c in zip(t.system, t.cost_per_mwh)]
        t.to_csv(root / "table7.csv", index=False)
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
