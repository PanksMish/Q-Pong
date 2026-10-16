#!/usr/bin/env python
"""Quantum layer in isolation on synthetic commitment-only UC-QUBO instances (Section VI-C).

Table V  (scalability): N_enc = 5..30. Reference energy: exhaustive enumeration up
         to 20 qubits; best of 100-restart simulated annealing and tabu search above.
Table VI (validation):  N_enc = 4..20 against exhaustive ground truth; optional
         Gurobi MIQP cross-check (--gurobi) at 16 and 20 qubits.

VQE runs from random parameters for all ``stage2.maxiter`` iterations.
Above 24 qubits the Aer simulator is used (set stage2.aer_device=GPU for cuQuantum).

Reported per size: T_VQE, E_VQE = |E - E_ref|/|E_ref|, r_approx = 1 - E_VQE,
S_shots / S_feas, P(opt), and - because E_VQE depends on the constant offset of
the Hamiltonian - also the offset-free normalized gap (E - E_ref)/(E_max - E_ref)
whenever E_max is available.

    python scripts/run_quantum_standalone.py --table V VI --seeds 0 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from _common import base_parser, config_for

from qpong.stage2_quantum.candidates import shot_feasible_share
from qpong.stage2_quantum.classical import exact_enumeration, gurobi_miqp, simulated_annealing, tabu_search
from qpong.stage2_quantum.instances import synthetic_uc
from qpong.stage2_quantum.qubo import build_qubo
from qpong.stage2_quantum.samplers import make_sampler, NumpySampler
from qpong.stage2_quantum.vqe import VQE
from qpong.utils import get_logger

log = get_logger("quantum")


def run_size(cfg, n, seed, use_gurobi=False):
    lam = cfg["stage2"]["penalties"]
    prob = synthetic_uc(n, seed=0)                       # instance fixed per size; seed = VQE seed
    q = build_qubo(prob, lam)
    rng = np.random.default_rng(seed)
    E_all = None
    if n <= 20:
        _, E0, E_all = exact_enumeration(q)
        e_ref, ref_kind = float(E0[0]), "exact"
    else:
        _, e_sa = simulated_annealing(q, sweeps=cfg["ablation"]["sa"]["sweeps"], restarts=100, rng=rng)
        _, e_tb = tabu_search(q, restarts=100, rng=rng)
        e_ref, ref_kind = min(e_sa, e_tb), "best_known(SA,tabu)"
    row = {"n_enc": n, "seed": seed, "e_ref": e_ref, "reference": ref_kind}
    if use_gurobi and n in (16, 20):
        try:
            _, eg, st = gurobi_miqp(q)
            row["e_gurobi"] = eg
        except Exception as ex:  # noqa: BLE001
            row["e_gurobi"] = f"unavailable: {ex.__class__.__name__}"
    sampler = make_sampler(cfg, n, seed=seed)
    res = VQE(cfg, sampler, rng).run(q, energies_all=E_all if isinstance(sampler, NumpySampler) else None)
    e = res.energy
    sf, ss = shot_feasible_share(prob, q, res.counts)
    row.update(t_vqe_s=res.time_s, e_vqe=e, residual=abs(e - e_ref) / abs(e_ref),
               approx_ratio=1 - abs(e - e_ref) / abs(e_ref), s_shots=ss, s_feas=sf,
               feas_rate=100 * sf / ss, n_iter=res.n_iter, n_params=len(res.theta))
    if E_all is not None:
        row["e_max"] = float(E_all.max())
        row["normalized_gap"] = (e - e_ref) / (E_all.max() - e_ref)
        row["delta_e"] = abs(e - e_ref)
        opt = int(np.argmin(E_all))
        if res.probabilities is not None:
            row["p_opt"] = float(res.probabilities[opt])
        else:
            row["p_opt"] = res.counts.get(opt, 0) / ss
        row["opt_is_most_probable"] = bool(max(res.counts, key=res.counts.get) == opt)
    return row


def main():
    ap = base_parser(__doc__)
    ap.set_defaults(system=["ieee30"])
    ap.add_argument("--table", nargs="+", default=["V", "VI"])
    ap.add_argument("--gurobi", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="override the register sizes")
    args = ap.parse_args()
    cfg = config_for(args.system[0], args)
    seeds = args.seeds or cfg["experiment"]["seeds"]
    out = Path(cfg["_repo_root"]) / cfg["experiment"]["results_dir"] / "quantum"
    out.mkdir(parents=True, exist_ok=True)
    sizes = {"V": [5, 10, 15, 20, 25, 30], "VI": [4, 8, 12, 16, 20]}
    for tab in args.table:
        rows = []
        for n in (args.sizes or sizes[tab]):
            for s in seeds:
                r = run_size(cfg, n, s, args.gurobi and tab == "VI")
                log.info("Table %s n=%d seed=%d: %s", tab, n, s,
                         {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
                rows.append(r)
                pd.DataFrame(rows).to_csv(out / f"table{tab}_raw.csv", index=False)
        df = pd.DataFrame(rows)
        num = df.select_dtypes("number").drop(columns=["seed"], errors="ignore")
        agg = num.groupby("n_enc").agg(["mean", "std"])
        agg.to_csv(out / f"table{tab}_summary.csv")
        print(agg.round(4).to_string())


if __name__ == "__main__":
    main()
