#!/usr/bin/env python
"""Aggregate run records into the paper's tables (CSV + LaTeX) under results/tables/.

Table IV  per-stage latency          (runs/main, Q-PONG)
Table IX  coalition-efficiency test  (runs/main: Q-PONG; runs/settlement: Deterministic, GFN-SV)
Table X   principal results          (runs/main, Q-PONG)
plus summary.csv with every metric for every method (Figs. 2-4, 6-10)
Tables V-VIII are written by run_quantum_standalone.py, run_noise.py and run_hardware.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpong.metrics import load_runs, optimality_gap, per_seed, stage_latency, summary  # noqa: E402


def fmt(m, s, d=2):
    return f"{m:.{d}f} ± {s:.{d}f}" if np.isfinite(m) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = Path(a.results)
    out = R / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs(R / "runs" / "main")
    if df.empty:
        sys.exit(f"no runs under {R / 'runs' / 'main'}")
    if (df.get("data_source") == "synthetic").any():
        print("WARNING: synthetic-data records present - these are smoke-test numbers, not results.")
    ps = optimality_gap(per_seed(df))
    ps.to_csv(out / "per_seed.csv", index=False)
    sm = summary(df)
    gap = ps.groupby(["method", "system", "penetration", "stress"])["gap_pct"].agg(["mean", "std"]).reset_index()
    sm = sm.merge(gap.rename(columns={"mean": "gap_pct", "std": "gap_pct_sd"}),
                  on=["method", "system", "penetration", "stress"], how="left")
    sm.to_csv(out / "summary.csv", index=False)

    # Table IV
    q = df[df.method == "Q-PONG"]
    if not q.empty:
        lat = stage_latency(q)
        lat.to_csv(out / "table4_latency.csv", index=False)
        print("\nTable IV (s)\n", lat.round(3).to_string(index=False))

    # Table X (first penetration level, normal stress)
    p0 = sorted(df.penetration.unique())[0]
    t10 = sm[(sm.method == "Q-PONG") & (sm.penetration == p0) & (sm.stress == "normal")].set_index("system")
    if not t10.empty:
        rows = {
            "Operating cost ($/MWh)": "cost_per_mwh", "Optimality gap (%)": "gap_pct", "Latency (s)": "latency_s",
            "First-pass PFR (S1, %)": "pfr_s1", "Issued-dispatch feasibility (S2, %)": "pfr_s2",
            "AC mismatch (p.u.)": "e_ac", "PINN error": "e_pinn", "Coalition error (E_coal)": "e_coal_max",
        }
        tab = pd.DataFrame({k: t10[v] for k, v in rows.items()}).T
        tab.to_csv(out / "table10_summary.csv")
        tab.to_latex(out / "table10_summary.tex", float_format="%.4g")
        print("\nTable X\n", tab.to_string())

    # Table IX
    rows = []
    for name, sub in (("main", df),):
        for (m, s), g in sub.groupby(["method", "system"]):
            if "e_coal" in g and g["e_coal"].notna().any():
                rows.append({"method": m, "system": s, "E_coal_max": g["e_coal"].max()})
    st = load_runs(R / "runs" / "settlement")
    if not st.empty and "e_coal" in st:
        for (m, s), g in st.groupby(["method", "system"]):
            if g["e_coal"].notna().any():
                rows.append({"method": m, "system": s, "E_coal_max": g["e_coal"].max()})
    if rows:
        t9 = pd.DataFrame(rows).drop_duplicates()
        t9.to_csv(out / "table9_ecoal.csv", index=False)
        print("\nTable IX\n", t9.to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
