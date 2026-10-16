"""Aggregation of JSONL run records into the metrics of Section V-E.

All rates are computed per seed and reported as mean +/- s.d. over seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_runs(root) -> pd.DataFrame:
    rows = []
    for f in Path(root).rglob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "times" in df:
        tdf = pd.json_normalize(df["times"]).add_prefix("time_")
        df = pd.concat([df.drop(columns=["times"]), tdf], axis=1)
    return df


def _ms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0) if len(x) else (np.nan, np.nan)


def per_seed(df: pd.DataFrame, keys=("method", "system", "penetration", "stress")):
    keys = list(keys)

    def agg(g):
        e = g["energy_mwh"].where(np.isfinite(g["cost"]))
        return pd.Series({
            "cost_per_mwh": g["cost"].sum(skipna=True) / max(e.sum(skipna=True), 1e-9),
            "pfr_s1": 100 * g["s1_feasible"].mean(),
            "pfr_s2": 100 * g["s2_feasible"].mean(),
            "latency_s": g["t_e2e"].mean(),
            "e_ac": g["e_ac"].mean() if "e_ac" in g else np.nan,
            "e_pinn": g["e_pinn"].mean() if "e_pinn" in g else np.nan,
            "nv_s2": g["s2_nv"].mean(), "ns_s2": g["s2_ns"].mean(),
            "nv_s1": g["s1_nv"].mean(), "ns_s1": g["s1_ns"].mean(),
            "fallback_rate": 100 * g["fallback"].astype(float).mean() if "fallback" in g else np.nan,
            "e_coal_max": g["e_coal"].max() if "e_coal" in g else np.nan,
            "gar": 100 * g["governance"].isin(["ADJUST", "REJECT"]).mean() if "governance" in g else np.nan,
            "n": len(g),
        })

    return df.groupby(keys + ["seed"]).apply(agg, include_groups=False).reset_index()


def summary(df: pd.DataFrame, keys=("method", "system", "penetration", "stress")) -> pd.DataFrame:
    ps = per_seed(df, keys)
    keys = list(keys)
    metrics = [c for c in ps.columns if c not in keys + ["seed"]]
    rows = []
    for k, g in ps.groupby(keys):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for m in metrics:
            r[m], r[m + "_sd"] = _ms(g[m])
        rows.append(r)
    return pd.DataFrame(rows)


def optimality_gap(ps: pd.DataFrame, ref="G-MINLP"):
    """Eq. (26) on normalized cost, per seed against the reference of the same seed."""
    keys = ["system", "penetration", "stress", "seed"]
    refc = ps[ps.method == ref].set_index(keys)["cost_per_mwh"]
    out = ps.copy()
    out["gap_pct"] = [100 * (c - refc.get(tuple(r[k] for k in keys), np.nan)) /
                      (abs(refc.get(tuple(r[k] for k in keys), np.nan)) + 1e-12)
                      for c, (_, r) in zip(out["cost_per_mwh"], out.iterrows())]
    return out


def stage_latency(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("time_")]
    return df.groupby(["method", "system"])[cols].mean().reset_index()


def coverage(df: pd.DataFrame, alphas=(0.2, 0.4, 0.6, 0.8), n_boot=1000, seed=0) -> pd.DataFrame:
    """Empirical coverage of the alpha-cut allocation interval (Fig. 16) with a cluster
    bootstrap over intervals (each interval = one cluster of its participants)."""
    from .stage4_settlement.gfn import alpha_cut, gaussian_coverage

    d = df.dropna(subset=["alloc_m"]) if "alloc_m" in df else df.iloc[0:0]
    rng = np.random.default_rng(seed)
    rows = []
    for a in alphas:
        hits = []
        for m, s, r in zip(d["alloc_m"], d["alloc_s"], d["alloc_real"]):
            lo, hi = alpha_cut(np.asarray(m), np.asarray(s), a)
            hits.append(((np.asarray(r) >= lo) & (np.asarray(r) <= hi)).astype(float))
        if not hits:
            continue
        per = np.array([h.mean() for h in hits])
        w = np.array([len(h) for h in hits])
        est = float((per * w).sum() / w.sum())
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(per), len(per))
            boots.append((per[i] * w[i]).sum() / w[i].sum())
        lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
        rows.append({"alpha": a, "coverage": 100 * est, "ci_lo": 100 * lo_ci, "ci_hi": 100 * hi_ci,
                     "analytic": 100 * float(gaussian_coverage(a)), "n_participant_intervals": int(w.sum())})
    return pd.DataFrame(rows)
