#!/usr/bin/env python
"""Regenerate Figs. 2-16 from the run records (PDF + PNG under results/figures/).

Each figure is produced only if its runs exist:
  runs/main        -> Figs. 2, 3, 4, 6, 7, 10      runs/stress or main (ieee118) -> Figs. 8, 9
  runs/settlement  -> Figs. 11, 12, 16             runs/ablation -> Figs. 13, 14, 15
  hardware/*/table8.csv + runs/noise/table7.csv -> Fig. 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpong.metrics import coverage, load_runs, optimality_gap, per_seed  # noqa: E402
from qpong.stage4_settlement.gfn import gaussian_coverage, width  # noqa: E402

ORDER = ["G-MINLP", "MOST", "QDUC", "PINN-OPF", "GFN-SV", "FBWM", "Q-PONG"]
SYS = ["ieee30", "ieee57", "ieee118"]
SYS_LABEL = {"ieee30": "IEEE 30-bus", "ieee57": "IEEE 57-bus", "ieee118": "IEEE 118-bus"}
COLORS = dict(zip(ORDER, ["#1f5fa6", "#2a9d6f", "#8b1e4a", "#c76fa8", "#5ab4e5", "#e6a417", "#d9550f"]))
HATCH = dict(zip(ORDER, ["", "//", "xx", "\\\\", "...", "oo", "**"]))


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def grouped_bars(ps, metric, ylabel, out, name, log=False):
    methods = [m for m in ORDER if m in set(ps.method)]
    systems = [s for s in SYS if s in set(ps.system)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.8 / max(len(methods), 1)
    for j, m in enumerate(methods):
        mu, sd = [], []
        for s in systems:
            v = ps[(ps.method == m) & (ps.system == s)][metric]
            mu.append(v.mean())
            sd.append(v.std(ddof=1) if len(v) > 1 else 0)
        x = np.arange(len(systems)) + (j - (len(methods) - 1) / 2) * w
        ax.bar(x, mu, w, yerr=sd, capsize=2, label=m, color=COLORS[m], hatch=HATCH[m], edgecolor="k", lw=0.4)
    ax.set_xticks(np.arange(len(systems)), [SYS_LABEL[s] for s in systems])
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = Path(a.results)
    out = R / "figures"

    main_df = load_runs(R / "runs" / "main")
    if not main_df.empty:
        p0 = sorted(main_df.penetration.unique())[0]
        d0 = main_df[(main_df.penetration == p0) & (main_df.stress == "normal")]
        ps = optimality_gap(per_seed(d0))
        grouped_bars(ps, "cost_per_mwh", "Operating cost, normalized ($/MWh, ±1 s.d.)", out, "fig02_cost")
        grouped_bars(ps[ps.method != "G-MINLP"], "gap_pct", "Optimality gap vs. G-MINLP (%, ±1 s.d.)", out, "fig03_gap")
        grouped_bars(ps, "latency_s", "End-to-end decision latency (s, ±1 s.d.)", out, "fig04_latency")
        grouped_bars(ps, "pfr_s1", "Physical feasibility (%) - first pass S1", out, "fig06_feasibility_s1")
        grouped_bars(ps, "pfr_s2", "Physical feasibility (%) - issued S2", out, "fig06_feasibility_s2")
        grouped_bars(ps, "e_ac", "AC power-flow mismatch (p.u., log)", out, "fig07_ac_mismatch", log=True)
        grouped_bars(ps[ps.method.isin(["PINN-OPF", "Q-PONG"])], "e_pinn", "Normalized prediction error",
                     out, "fig10_pinn_error")

    for name, src in (("stress", R / "runs" / "stress"), ("main", R / "runs" / "main")):
        df = load_runs(src)
        if df.empty or "ieee118" not in set(df.system):
            continue
        d = df[(df.system == "ieee118") & (df.stress == "normal")]
        for metric, s1, lab, fig in (("s2_nv", "s1_nv", "Voltage", "fig08"), ("s2_ns", "s1_ns", "Thermal", "fig09")):
            f, ax = plt.subplots(figsize=(5, 3.4))
            for m in [m for m in ORDER if m in set(d.method)]:
                g = d[d.method == m].groupby("penetration")[metric].mean()
                ax.plot(100 * g.index, g.values, marker="o", label=f"{m} (issued)", color=COLORS[m])
                if m == "Q-PONG":
                    g1 = d[d.method == m].groupby("penetration")[s1].mean()
                    ax.plot(100 * g1.index, g1.values, ls="--", marker="*", mfc="none", color=COLORS[m],
                            label="Q-PONG first pass (rejected by gate)")
            ax.set_xlabel("Renewable penetration (%)")
            ax.set_ylabel(f"{lab} violations / dispatch (IEEE 118-bus)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            save(f, out, f"{fig}_{lab.lower()}_violations")
        break

    st = load_runs(R / "runs" / "settlement")
    if not st.empty:
        q = st[st.method == "Q-PONG"]
        # Fig. 11: analytic width for each method's mean relative spread
        f, ax = plt.subplots(figsize=(5, 3.4))
        al = np.linspace(0.01, 1, 200)
        for m in ("GFN-SV", "Q-PONG"):
            g = st[st.method == m]
            if g.empty or "sigma_rel_mean" not in g:
                continue
            sbar = g["sigma_rel_mean"].mean()
            ax.plot(al, width(sbar, al), color=COLORS[m], label=f"{m} (σ̄={sbar:.4f})")
            ax.plot([0.2, 0.4, 0.6, 0.8, 1.0], width(sbar, np.array([0.2, 0.4, 0.6, 0.8, 1.0])), "o", color=COLORS[m])
        ax.set_xlabel("α-cut level")
        ax.set_ylabel(r"$W(\alpha)=2\bar\sigma\sqrt{-2\ln\alpha}$")
        ax.legend()
        ax.grid(alpha=0.3)
        save(f, out, "fig11_alpha_width")
        # Fig. 12: GAR by alpha (one run per alpha level)
        rows = []
        for (al_used, seed), g in q.groupby(["alpha_used", "seed"]):
            rows.append({"alpha": al_used, "seed": seed, "gar": 100 * g["governance"].isin(["ADJUST", "REJECT"]).mean()})
        if rows:
            gar = pd.DataFrame(rows).groupby("alpha")["gar"].agg(["mean", "std"])
            f, ax = plt.subplots(figsize=(5, 3.4))
            ax.errorbar(gar.index, gar["mean"], yerr=gar["std"], marker="*", color=COLORS["Q-PONG"], capsize=3)
            ax.set_xlabel("α-cut level")
            ax.set_ylabel("Governance adjustment rate (%, ±1 s.d.)")
            ax.grid(alpha=0.3)
            save(f, out, "fig12_gar")
        # Fig. 16: coverage
        cov = coverage(q)
        if not cov.empty:
            cov.to_csv(out / "fig16_coverage.csv", index=False)
            f, ax = plt.subplots(figsize=(5, 3.4))
            ax.plot(al, 100 * gaussian_coverage(al), color=COLORS["Q-PONG"], label=r"expected $1-2\Phi(-\sqrt{-2\ln\alpha})$")
            ax.errorbar(cov.alpha, cov.coverage, yerr=[cov.coverage - cov.ci_lo, cov.ci_hi - cov.coverage],
                        fmt="*", color="k", capsize=3, label="empirical (95% cluster bootstrap)")
            ax.set_xlabel("α-cut level")
            ax.set_ylabel("Coverage of interval (%)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            save(f, out, "fig16_coverage")

    ab = load_runs(R / "runs" / "ablation")
    if not ab.empty:
        ps = optimality_gap(per_seed(ab))
        order = ["Fixed", "Classical (SA)", "Exact", "Data-only", "No-Verif.", "Deterministic", "No-HITL", "Q-PONG"]
        ps = ps[ps.method.isin(order)]
        for metric, lab, fig in (("gap_pct", "Optimality gap (%)", "fig13_ablation_gap"),
                                 ("pfr_s1", "First-pass physical feasibility (%)", "fig14_ablation_pfr"),
                                 ("latency_s", "End-to-end latency (s)", "fig15_ablation_latency")):
            g = ps.groupby("method")[metric].mean().reindex(order).dropna()
            f, ax = plt.subplots(figsize=(6, 3.4))
            ax.bar(range(len(g)), g.values, color=["#d9550f" if m == "Q-PONG" else "#999" for m in g.index],
                   edgecolor="k", lw=0.4)
            for i, v in enumerate(g.values):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            if metric == "pfr_s1":
                s2 = ps.groupby("method")["pfr_s2"].mean().reindex(g.index)
                ax.plot(range(len(g)), s2.values, "_", ms=20, color="#d9550f", label="issued (S2)")
                ax.legend(fontsize=7)
            ax.set_xticks(range(len(g)), g.index, rotation=30, ha="right")
            ax.set_ylabel(lab)
            ax.grid(axis="y", alpha=0.3)
            save(f, out, fig)

    t7 = R / "runs" / "noise" / "table7.csv"
    t8 = sorted(Path("hardware").glob("*/table8.csv"))
    t8 = [p for p in t8 if not p.parent.name.startswith("dryrun_")]
    if t7.exists() and t8:
        n = pd.read_csv(t7)
        h = pd.read_csv(t8[0])
        systems = [s for s in SYS if s in set(h.system)]
        f, axs = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True)
        series = [("noiseless", lambda s: n[(n.system == s) & (n["mode"] == "noiseless")]),
                  ("noise model", lambda s: n[(n.system == s) & (n["mode"] == "noise")]),
                  ("ibm raw", lambda s: h[(h.system == s) & (h["mode"] == "raw")]),
                  ("ibm mitigated", lambda s: h[(h.system == s) & (h["mode"] == "mitigated")])]
        for k, (lab, sel) in enumerate(series):
            x = np.arange(len(systems)) + (k - 1.5) * 0.2
            axs[0].bar(x, [sel(s)["accuracy"].mean() for s in systems], 0.2, label=lab)
            axs[1].bar(x, [sel(s)["feas_share"].mean() for s in systems], 0.2, label=lab)
        axs[0].set_ylabel("Ground-state accuracy (%)")
        axs[1].set_ylabel(r"$S_{feas}/S_{shots}$ (%)")
        axs[1].set_xticks(range(len(systems)), [SYS_LABEL[s] for s in systems])
        axs[0].legend(fontsize=7, ncol=4)
        save(f, out, "fig05_hardware")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
