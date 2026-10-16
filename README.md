# Q-PONG

**Quantum-Physics-Informed Operations and Negotiated Governance** — a hybrid quantum-classical
framework for adaptive unit commitment and risk-aware smart-grid operation.

Reference implementation of

> P. Mishra, S. Okuhara, T. Saha, S. Parab and Y. Singh, *"Q-PONG: A Hybrid Quantum-Physics-Informed
> Framework for Adaptive Unit Commitment and Risk-Aware Smart Grid Operation"* (submitted).

```
 ξ_t ──► Stage I ──► w_t ──► Stage II ──► U_t ──► Stage III ──► x*_t ──► Stage IV ──► φ̃_t , g_t
        context-       VQE on the        PINN prediction     Gaussian-fuzzy Shapley
        adaptive       reduced UC        + independent AC    settlement + α-cut
        fuzzy BWM      encoding          verification gate   governance
          ▲                                + fallback                    │
          └──────────────── feedback e_t (eq. 21) ───────────────────────┘
```

| Paper | Code |
|---|---|
| Eq. (1), (8), (31)–(33), Alg. 1 — context-adaptive FBWM | `qpong/stage1_fbwm/` |
| Eq. (9)–(14), Alg. 2 — reduced UC QUBO, Ising map, EfficientSU2 + SPSA VQE, candidates | `qpong/stage2_quantum/` |
| Eq. (15)–(18), (34)–(40), Alg. 3 — PINN, verification gate, repair, fallback | `qpong/stage3_pinn/`, `qpong/grid/` |
| Eq. (3)–(5), (19)–(20), (41)–(42) — GFNs, fuzzy Shapley, governance policy | `qpong/stage4_settlement/` |
| Eq. (6), (21), Alg. 4 — closed loop | `qpong/pipeline.py` |
| Eq. (23), (25)–(29) — metrics | `qpong/metrics.py` |
| Sec. V-B — data, splits, eq. (24), stress classes | `qpong/data/` |
| Sec. V-D — baselines G-MINLP, MOST, QDUC, PINN-OPF, FBWM, GFN-SV | `qpong/baselines/` |
| Sec. V-F — ablations | `qpong/experiments.py` (`ABLATIONS`) |
| Sec. VI-D/E — noise model, IBM Heron execution | `qpong/stage2_quantum/noise.py`, `hardware.py` |
| Tables IV–X, Figs. 2–16 | `scripts/` |

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/PanksMish/Q-PONG.git && cd Q-PONG
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # core + IBM Runtime + pvlib + dev tools
# optional: pip install gurobipy (licence needed) for the Gurobi G-MINLP back-end
# optional: pip install qiskit-aer-gpu for cuQuantum-accelerated simulation (stage2.aer_device=GPU)
pytest -q                        # 24 unit tests, ~10 s
```

The AC power flow / OPF engine is **PYPOWER**, the Python port of MATPOWER, running the original MATPOWER
case files `case30`, `case57`, `case118` (not the PGLib-OPF variants). Its OPF reproduces MATPOWER's
reference optima (e.g. 576.89 $/h for case30; see `tests/test_grid_and_fbwm.py`).

## Data

| Source | Use | How to obtain |
|---|---|---|
| Pecan Street Dataport (Austin, 25 homes, 15 min, 2018) | demand, rooftop PV | licensed — export manually, see `data/README.md` |
| NREL NSRDB PSM3 (Austin, 2018) | GHI/DNI → PV availability | `python scripts/download_data.py nsrdb --api-key … --email …` |
| ERCOT DAM/RTM settlement-point prices, 4 hubs | Stage I price deviation, Stage IV settlement | `python scripts/download_data.py ercot` (prints the report pages) |
| EPA eGRID 2023, ERCT | carbon factor | `python scripts/download_data.py egrid` (prints the page) |

Then `python scripts/prepare_data.py --rebuild` builds `data/processed/profiles_2018.csv.gz` and prints
a scenario check (share of intervals whose net load exceeds the capacity of each case). **Run it and read it
before the experiments** — see `docs/IMPLEMENTATION_NOTES.md` §2.

Without the licensed data, `--smoke` (or `data.source=synthetic`) uses clearly-labelled synthetic profiles
for testing only. Every record carries `data_source`; `make_tables.py` warns on synthetic records.

## Reproducing the paper

```bash
python scripts/prepare_data.py --rebuild
python scripts/train_pinn.py --variant qpong data_only pinn_opf --n-jobs 8      # offline, not in latency
python scripts/run_quantum_standalone.py --table V VI                           # Tables V, VI
python scripts/run_comparison.py --n-jobs 16                                    # Figs 2-4, 6-10, Tables IV, X
python scripts/run_comparison.py --system ieee118 --tag stress \
       --stress normal high_demand low_renewables combined                      # robustness
python scripts/run_ablation.py                                                  # Figs 13-15 (57-bus)
python scripts/run_settlement.py                                                # Table IX, Figs 11, 12, 16
python scripts/run_noise.py                                                     # Table VII
python scripts/run_hardware.py train|submit|replay --backend ibm_kingston       # Table VIII, Fig 5
python scripts/make_tables.py && python scripts/make_figures.py
```

`scripts/run_all.sh` chains everything. The full grid is 3 systems × 3 penetrations × 7 methods ×
10 seeds × 30 days × 96 intervals; split it with `--days`, `--seeds`, `--intervals`, `--n-jobs`.
Runs resume per day. Results go to `results/runs/<tag>/<system>/<method>/*.jsonl`
(one JSON record per interval), tables to `results/tables/`, figures to `results/figures/`.

Indicative CPU cost per interval (this implementation): 30/57-bus Q-PONG 1–7 s; 118-bus VQE on the
19-qubit register ≈ 0.2 s per circuit evaluation with Aer CPU (≈ 1–2 min per cold-started VQE);
use `stage2.aer_device=GPU` for cuQuantum. G-MINLP (enumeration back-end) 5–20 s on 30/57-bus.

### IBM Quantum hardware

```bash
export QISKIT_IBM_TOKEN=...   # optional QISKIT_IBM_INSTANCE
python scripts/run_hardware.py train  --system ieee30 ieee57 ieee118
python scripts/run_hardware.py submit --system ieee30 ieee57 ieee118 --backend ibm_kingston
python scripts/run_hardware.py replay --system ieee30 ieee57 ieee118
```

Job IDs, raw counts, transpiled gate counts and the calibration snapshot are written to
`hardware/<run-id>/` — commit that folder to make the hardware run auditable. `--dry-run`
exercises the whole workflow on `FakeKingston` (local simulation; never report those numbers).

## Configuration

`configs/default.yaml` holds Table III and every other parameter; `configs/ieee*.yaml` hold the
encodings. Any key can be overridden: `--set stage2.maxiter=100 stage4.alpha_default=0.8`.
Parameters the paper does not fix are marked `MODELING CHOICE` in the YAML and listed in
`docs/IMPLEMENTATION_NOTES.md`.

## Repository layout

```
configs/            default.yaml (Table III) + per-system encodings
qpong/              package: data, grid, stage1_fbwm, stage2_quantum, stage3_pinn,
                    stage4_settlement, baselines, pipeline, experiments, metrics
scripts/            data download/preparation, training, every experiment, tables, figures
tests/              unit tests (QUBO/Ising, simulator vs Qiskit, OPF vs MATPOWER, Shapley, alpha-cuts, FBWM)
docs/               implementation notes and reproduction checklist
data/               raw/ (not committed) and processed/
hardware/           IBM Runtime job records (committed after a real run)
results/            generated (not committed by default)
```

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`). Pecan Street data are not redistributed.
