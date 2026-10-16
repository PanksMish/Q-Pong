#!/usr/bin/env python
"""Settlement and governance (Table IX, Figs. 11, 12, 16).

Runs Q-PONG at every alpha level of ``stage4.alpha_levels`` (the governance
adjustment rate depends on alpha), plus Deterministic and GFN-SV, on the
given system (default 57-bus, held-out test scenarios).

    python scripts/run_settlement.py --n-jobs 8
"""
from __future__ import annotations

from _common import base_parser, config_for, smoke_defaults

from qpong.experiments import plan_jobs, run_jobs


def main():
    ap = base_parser(__doc__)
    ap.set_defaults(system=["ieee57"])
    args = smoke_defaults(ap.parse_args())
    jobs = []
    for system in args.system:
        base = config_for(system, args)
        for a in base["stage4"]["alpha_levels"]:
            args_a = list(args.set) + [f"stage4.alpha_default={a}"]
            cfg = config_for(system, type("A", (), {"set": args_a, "smoke": args.smoke})())
            jobs += plan_jobs(cfg, ["Q-PONG"], f"{args.tag or 'settlement'}/alpha{int(a * 10)}",
                              [base["data"]["penetrations"][0]], ["normal"], args.seeds, args.days, args.intervals)
        jobs += plan_jobs(base, ["Deterministic", "GFN-SV"], f"{args.tag or 'settlement'}/other",
                          [base["data"]["penetrations"][0]], ["normal"], args.seeds, args.days, args.intervals)
    print(f"{len(jobs)} jobs")
    run_jobs(jobs, args.n_jobs)


if __name__ == "__main__":
    main()
