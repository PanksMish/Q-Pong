#!/usr/bin/env python
"""Main comparison (Figs. 2-4, 6-10, Tables IV and X) and the penetration / stress sweeps (Figs. 8-9).

    python scripts/run_comparison.py --system ieee30 ieee57 ieee118 --n-jobs 16
    python scripts/run_comparison.py --system ieee118 --stress normal high_demand low_renewables combined \
        --tag stress
"""
from __future__ import annotations

from _common import base_parser, config_for, smoke_defaults

from qpong.experiments import METHODS, plan_jobs, run_jobs


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--penetration", type=float, nargs="*", default=None)
    ap.add_argument("--stress", nargs="+", default=["normal"])
    args = smoke_defaults(ap.parse_args())
    jobs = []
    for system in args.system:
        cfg = config_for(system, args)
        jobs += plan_jobs(cfg, args.methods, args.tag or "main", args.penetration, args.stress, args.seeds,
                          args.days, args.intervals)
    print(f"{len(jobs)} jobs")
    run_jobs(jobs, args.n_jobs)


if __name__ == "__main__":
    main()
