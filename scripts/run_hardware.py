#!/usr/bin/env python
"""Execution on IBM Quantum hardware (Section VI-E, Table VIII, Fig. 5).

Three phases:
  1. ``train``  : noiseless Q-PONG run of the peak-load test day, recording the
                  trained parameters theta*_t and the Ising Hamiltonian of every
                  interval (96 intervals for 30-bus, the 24 hourly intervals for
                  57- and 118-bus).
  2. ``submit`` : transpile (optimization level 3, lowest-error layout) and run
                  every circuit raw and mitigated (Sampler: XY4 DD + gate/measure
                  twirling; Estimator: TREX) with 8192 shots. Job IDs, raw counts
                  and the calibration snapshot go to hardware/<run_id>/.
  3. ``replay`` : rerun the pipeline with the hardware counts in place of VQE
                  sampling (candidate selection, verification, fallback and
                  settlement unchanged) and write Table VIII.

    export QISKIT_IBM_TOKEN=...            # or QiskitRuntimeService.save_account(...)
    python scripts/run_hardware.py train  --system ieee30 ieee57 ieee118
    python scripts/run_hardware.py submit --system ieee30 ieee57 ieee118 --backend ibm_kingston
    python scripts/run_hardware.py replay --system ieee30 ieee57 ieee118

``--dry-run`` uses FakeKingston (local noisy simulation) to test the workflow;
dry-run outputs are NOT hardware results and are written under hardware/dryrun_*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _common import base_parser, config_for

from qpong.data import day_splits, load_profiles, make_scenario
from qpong.experiments import _jsonable, _load_model
from qpong.grid import GridModel
from qpong.pipeline import QPONG
from qpong.utils import get_logger, set_seed

log = get_logger("hardware")


def peak_day(grid, prof, meta, cfg):
    best, day = -1, None
    for d in day_splits()["test"]:
        sc = make_scenario(grid, prof, meta, cfg, d, cfg["data"]["penetrations"][0])
        pk = sc.pd.sum(1).max()
        if pk > best:
            best, day = pk, d
    return day


def intervals_for(system):
    return list(range(96)) if system == "ieee30" else list(range(0, 96, 4))


def run_dir(cfg, args):
    base = Path(cfg["_repo_root"]) / "hardware"
    return base / (("dryrun_" if args.dry_run else "") + (args.run_id or "kingston"))


def phase_train(system, cfg, args):
    cfg["stage2"]["record_theta"] = True
    set_seed(0)
    grid = GridModel.from_config(cfg)
    prof, meta = load_profiles(cfg)
    day = peak_day(grid, prof, meta, cfg)
    sc = make_scenario(grid, prof, meta, cfg, day, cfg["data"]["penetrations"][0])
    q = QPONG(grid, cfg, _load_model(grid, cfg, "qpong", 0), np.random.default_rng(0))
    recs = q.run_day(sc, args.intervals or intervals_for(system))
    out = run_dir(cfg, args) / system
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "noiseless.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(_jsonable(r)) + "\n")
    log.info("%s: trained %d intervals of day %d", system, len(recs), day)


def phase_submit(system, cfg, args):
    from qpong.stage2_quantum.hardware import get_backend, run_on_hardware
    from qpong.stage2_quantum.qubo import to_sparse_pauli_op

    d = run_dir(cfg, args) / system
    recs = [json.loads(line) for line in (d / "noiseless.jsonl").read_text().splitlines()]
    backend, service = get_backend(args.backend, args.dry_run)
    jobs = []
    for r in recs:
        if "theta" not in r:
            continue
        op = to_sparse_pauli_op(r["ising_kappa"], np.array(r["ising_h"]), np.array(r["ising_J"]))
        jobs.append({"key": f"{system}_t{r['t']}", "n": len(r["ising_h"]), "theta": r["theta"], "hamiltonian": op})
    res = run_on_hardware(jobs, backend, d, shots=cfg["stage2"]["shots"], reps=cfg["stage2"]["reps"],
                          service=service)
    log.info("%s: %d circuits executed on %s", system, len(res), backend.name)


def phase_replay(system, cfg, args):
    d = run_dir(cfg, args) / system
    recs = {json.loads(x)["t"]: json.loads(x) for x in (d / "noiseless.jsonl").read_text().splitlines()}
    jobs = {j["key"]: j for j in json.loads((d / "jobs.json").read_text())["jobs"]}
    counts = json.loads((d / "raw_counts.json").read_text())
    grid = GridModel.from_config(cfg)
    prof, meta = load_profiles(cfg)
    day = next(iter(recs.values()))["day"]
    sc = make_scenario(grid, prof, meta, cfg, day, cfg["data"]["penetrations"][0])
    rows = []
    for tag in ("raw", "mitigated"):
        provider = lambda t, tag=tag: {int(k): v for k, v in counts[f"{system}_t{t}"][tag].items()}  # noqa: E731
        q = QPONG(grid, cfg, _load_model(grid, cfg, "qpong", 0), np.random.default_rng(0), counts_provider=provider)
        out = q.run_day(sc, sorted(recs))
        with open(d / f"replay_{tag}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(_jsonable(r)) + "\n")
        acc = [100 * (1 - abs(jobs[f"{system}_t{t}"][f"energy_{tag}"] - recs[t]["e_exact"]) / abs(recs[t]["e_exact"]))
               for t in sorted(recs)]
        df = pd.DataFrame(out)
        e = df["energy_mwh"].where(np.isfinite(df["cost"]))
        rows.append({"system": system, "mode": tag, "n_enc": grid.n_enc, "intervals": len(out),
                     "twoq_gates": jobs[f"{system}_t{sorted(recs)[0]}"]["twoq_gates"],
                     "depth": jobs[f"{system}_t{sorted(recs)[0]}"]["depth"],
                     "accuracy": float(np.mean(acc)),
                     "feas_share": float(100 * df["s_feas"].sum() / df["s_shots"].sum()),
                     "fallback": int(df["fallback"].sum()),
                     "issued_feasible": float(100 * df["s2_feasible"].mean()),
                     "cost_per_mwh": float(df["cost"].sum() / e.sum())})
    noiseless = pd.DataFrame(list(recs.values()))
    e = noiseless["energy_mwh"].where(np.isfinite(noiseless["cost"]))
    c0 = noiseless["cost"].sum() / e.sum()
    for r in rows:
        r["cost_increase_pct"] = 100 * (r["cost_per_mwh"] / c0 - 1)
        r["noiseless_accuracy"] = float(np.mean(100 * (1 - (noiseless["vqe_energy"] - noiseless["e_exact"]).abs()
                                                       / noiseless["e_exact"].abs())))
    return rows


def main():
    ap = base_parser(__doc__)
    ap.add_argument("phase", choices=["train", "submit", "replay"])
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = []
    for system in args.system:
        cfg = config_for(system, args)
        if args.phase == "train":
            phase_train(system, cfg, args)
        elif args.phase == "submit":
            phase_submit(system, cfg, args)
        else:
            rows += phase_replay(system, cfg, args)
    if rows:
        t = pd.DataFrame(rows)
        out = run_dir(config_for(args.system[0], args), args) / "table8.csv"
        t.to_csv(out, index=False)
        print(t.round(3).to_string(index=False))
        if args.dry_run:
            print("\nDRY RUN (FakeKingston simulation) - not hardware results.", file=sys.stderr)


if __name__ == "__main__":
    main()
