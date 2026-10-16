#!/usr/bin/env python
"""End-to-end smoke test on SYNTHETIC data (CI): trains tiny PINNs, runs every method
and ablation for a few intervals, aggregates tables and figures under results/smoke/.
Numbers produced here are meaningless as results; they only show the code runs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ["--smoke"]


def run(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


if __name__ == "__main__":
    systems = sys.argv[1:] or ["ieee30"]
    run("scripts/train_pinn.py", "--system", *systems, *S)
    run("scripts/run_comparison.py", "--system", *systems, *S)
    run("scripts/run_ablation.py", "--system", *systems, *S)
    run("scripts/run_settlement.py", "--system", *systems, *S, "--intervals", "8", "40")
    run("scripts/make_tables.py", "--results", "results/smoke")
    run("scripts/make_figures.py", "--results", "results/smoke")
    print("smoke test OK")
