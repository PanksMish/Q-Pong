# Implementation notes

This file records (1) every place where the implementation had to choose something the paper does not
specify, and (2) points where the paper text and this code base should be reconciled before submission.
All choices are configurable in `configs/default.yaml` (search for `MODELING CHOICE`).

## 1. Modeling choices

| Topic | Choice in this code | Config key |
|---|---|---|
| AC engine | PYPOWER (Python port of MATPOWER, MIPS OPF, Newton PF) on MATPOWER case files | — |
| Min stable generation | Pmin = 30 % of Pmax for encoded units (MATPOWER cases have Pmin = 0, which makes UC trivial) | `grid.pmin_fraction` |
| No-load / start-up cost | c_i = 0.1·b_i·Pmax_i, s_i = 1 h·b_i·Pmax_i | `grid.noload_cost_fraction`, `grid.startup_cost_hours` |
| Ramping, min up/down | 25 % Pmax / interval, 4 intervals | `grid.*` |
| Unrated branches | MATPOWER case57/case118 carry no thermal ratings (RATE_A = 0 / 9900). Unrated branches get 1.5 × base-case OPF flow (floor 50 MVA); without this no thermal violation can occur on those systems | `grid.thermal_ratings`, `grid.derived_rating_*` |
| 118-bus encoding | 19 zones anchored on the 19 units with Pmax > 100 MW (incl. slack at bus 69); the 35 Pmax = 100 MW units are synchronous condensers (P = 0) at fixed voltage | `configs/ieee118.yaml` |
| P_i with K_i = 0 | committed encoded unit evaluated at nominal loading clip(0.75·Pmax, Pmin, Pmax) in the QUBO | `grid.target_loading` |
| Criterion normalization | f_k divided by its value with all encoded units committed (f_min = 0) | — |
| Network-security proxy R_t | Σ_i ρ_i P_i, ρ_i = loading²-weighted mean |PTDF| of the unit's bus (previous interval) | — |
| Emission factors | uniform eGRID ERCT rate for every unit (as in the paper) | `data.egrid_*` |
| Renewable penetration | share of annual demand energy supplied by PV | — |
| PV placement | largest 30 % of load buses, proportional to base load | `data.pv_buses_fraction` |
| PV curtailment | PV curtailed so net load ≥ 20 % of demand (otherwise midday net load is negative at 40–60 %) | `data.min_netload_fraction` |
| Forecasts | intraday = persistence (t−1); day-ahead = same interval of previous day | `data.forecast` |
| Frequency f_t | not in any data source; derived from net-load forecast error with an aggregate response of 1 × load per Hz | `stage1.freq_response_per_hz` |
| FBWM preferences, context sensitivities | baseline TFNs and γ vectors in YAML | `stage1.*` |
| BWM program (33) | SLSQP on the ratio form, warm-started from the linear BWM LP | — |
| SPSA | learning-rate calibration as in Qiskit (first step 2π/10); warm-started runs stop on convergence (tol 1e-4, patience 30) | `stage2.spsa.*` |
| Candidate pre-screen | candidates violating adequacy or min up/down are dropped before the gate (kept if none remain) | — |
| Gate | V_phys (eq. 16) on the PINN prediction ≤ τ_phys = 0.05 **and** AC power flow of the issued set-points under the realized state with N_V = N_S = N_G = N_bal = 0 at 1e-4 p.u. | `stage3.tau_phys`, `grid.verification_tol_pu` |
| Repair | one projection: clip P to limits, rescale toward net load + 2 % losses, clip V set-points | `stage3.max_repair_attempts` |
| Fallback | AC-OPF over up to 10 ranked candidates, then a merit-order commitment, then all units | — |
| Coalition values (41)–(42) | revenue at RT price, quadratic generation cost, 2 $/MWh network charge, imbalance cost 1.2 × price × shortfall vs. capacity share of net load; spreads from rolling forecast-error s.d. | `objectives.*`, `stage4.*` |
| Allocation spread σ̄ for Fig. 11 | mean relative spread σ_φ/|m_φ| | — |
| Coverage (Fig. 16) | realized payoff = Shapley allocation recomputed ex post with realized load, PV and price | `stage4.eval_coverage` |
| ADJUST action | α ← min(1, α + 0.2), intervals recomputed, then approved | `stage4.alpha_adjust_step` |
| REJECT | next admitted candidate, at most 3 times, then approve | `stage4.max_governance_rejects` |
| Operating cost | (generation + start-up cost) / served net energy, $/MWh | — |
| Optimality gap | eq. (26) on normalized operating cost, per seed vs. G-MINLP of the same scenario (G-MINLP itself optimizes the scalarized objective F) | — |

## 2. Points to reconcile with the manuscript

1. **Hardware dates.** Section VI-E states the circuits were executed on `ibm_kingston` on
   20–24 October 2026. Make sure the reported Table VIII / Fig. 5 numbers come from that run and that
   `hardware/<run-id>/` (job IDs, raw counts, calibration snapshot) is committed, since the Code and
   Data Availability statement says the repository contains them.
2. **VQE quality.** With the stated configuration (EfficientSU2, L = 3, SPSA 300 iterations, 8192 shots,
   random start), this implementation's VQE on its UC-QUBO instances stays well above the ground state
   (≈ 10 % of the spectrum width at 6–12 qubits in our checks) with small P(opt). Tables V–VI must be
   regenerated with `run_quantum_standalone.py`, and the text in the Abstract, VI-C and VII adjusted to
   whatever it produces.
3. **"Accuracy" depends on the Hamiltonian offset.** 1 − |E − E_ref|/|E_ref| changes with any constant
   added to H (penalty constants, normalization), so a value near 99 % can be produced by the offset
   alone. `run_quantum_standalone.py` also reports the offset-free normalized gap
   (E − E_ref)/(E_max − E_ref) and P(opt); consider reporting those.
4. **Demand mapping (eq. 24).** D_t = D0·L_t/median(L) can push evening peaks of residential load above
   the installed capacity of case30 (189 MW base load vs 335 MW capacity) and case57. Uniform +10 %
   load already makes the MATPOWER case30/57 AC-OPF infeasible with all units on (verified with two
   independent MIPS implementations), which affects the +15 % stress class. Check the output of
   `prepare_data.py`; `data.demand_normalization: peak` with `peak_load_factor` is provided as an option.
5. **MOST.** `qpong/baselines/most.py` re-implements a MOST-style rolling two-stage stochastic DC UC in
   Python; it is not MATPOWER MOST. Either run MOST in MATLAB/Octave or describe the baseline as
   MOST-style.
6. **G-MINLP.** The Gurobi back-end (rectangular-coordinate nonconvex MIQCP) is implemented but was not
   executed in development (no licence available there). Without Gurobi, the enumeration back-end is
   exact over the encoded commitment space for N_enc ≤ 10 (30/57-bus) and a screened, non-certified
   search for 118-bus; records carry `certified` and `backend`.
7. **PINN-OPF.** Implemented with data + AC-residual + limit penalties; the KKT-residual term of the
   original paper is not included.
8. **Thermal ratings** of case57/case118 (see table above) and the 118-bus zone map must match what the
   manuscript describes.
9. **No numbers are shipped.** `results/` is empty in the repository; every table and figure must be
   regenerated from real data before it is reported. Smoke-test outputs (`results/smoke/`, synthetic
   data) are meaningless as results.
