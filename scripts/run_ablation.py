#!/usr/bin/env python
"""Ablations on the IEEE 57-bus system (Section V-F, Figs. 13-15).

    python scripts/run_ablation.py --n-jobs 8
"""
from __future__ import annotations

from _common import base_parser, config_for, smoke_defaults

from qpong.experiments import ABLATIONS, plan_jobs, run_jobs


def main():
    ap = base_parser(__doc__)
    ap.set_defaults(system=["ieee57"])
    ap.add_argument("--variants", nargs="+", default=list(ABLATIONS) + ["Q-PONG", "G-MINLP"])
    args = smoke_defaults(ap.parse_args())
    jobs = []
    for system in args.system:
        cfg = config_for(system, args)
        jobs += plan_jobs(cfg, args.variants, args.tag or "ablation", [cfg["data"]["penetrations"][0]],
                          ["normal"], args.seeds, args.days, args.intervals)
    print(f"{len(jobs)} jobs")
    run_jobs(jobs, args.n_jobs)


if __name__ == "__main__":
    main()
