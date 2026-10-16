"""Shared CLI helpers for the scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qpong.config import load_config  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--system", nargs="+", default=["ieee30", "ieee57", "ieee118"])
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. stage2.maxiter=100 data.source=synthetic")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--days", type=int, nargs="*", default=None, help="0-based day indices (default: 30 test days)")
    p.add_argument("--intervals", type=int, nargs="*", default=None, help="interval indices (default: all 96)")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--tag", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="synthetic data, 1 seed, 1 day, 4 intervals, short training (CI only)")
    return p


SMOKE = ["data.source=synthetic", "data.processed_file=data/processed/synthetic_2018",
         "experiment.results_dir=results/smoke", "stage3.max_epochs=30", "stage3.early_stop_patience=15",
         "stage3.n_train_samples=200", "stage2.maxiter=60", "stage2.warm_maxiter=40",
         "stage4.n_permutations=100", "baselines.gminlp.enumerate_top_k=8",
         "baselines.day_ahead_milp_time_limit_s=60"]


def config_for(system: str, args) -> dict:
    ov = (SMOKE if getattr(args, "smoke", False) else []) + list(args.set)
    return load_config(system, overrides=ov)


def smoke_defaults(args):
    if args.smoke:
        args.seeds = args.seeds or [0]
        args.days = args.days or [11]
        args.intervals = args.intervals or [0, 32, 56, 80]
    return args
