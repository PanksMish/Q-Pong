"""Q-PONG closed loop (Algorithm 4, eqs. 6 and 21).

    QPONG = GFSG o P_PINN o Q_VQE o FBWM :  xi_t -> w_t -> U_t -> x*_t -> phi~_t

One ``QPONG`` object runs one day (96 intervals) of one scenario and keeps
the inter-interval state: previous commitment, up/down timers, warm-start
VQE parameters, the last measured voltages/loadings and the feedback e_t.
Per-stage wall-clock times are recorded for Table IV.

Ablation switches (configs: ``ablation.*``):
  fixed_weights, quantum_solver (vqe|sa|exact), pinn_physics (selects the
  data-only model file), verification, fuzzy_settlement, hitl.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid.acpf import ac_mismatch
from .grid.acopf import solve_acopf
from .stage1_fbwm import FBWM, context_vector, feedback_update
from .stage2_quantum.candidates import rank_candidates, shot_feasible_share
from .stage2_quantum.classical import exact_enumeration, simulated_annealing
from .stage2_quantum.qubo import build_qubo, hard_feasible
from .stage2_quantum.samplers import NumpySampler, make_sampler
from .stage2_quantum.vqe import VQE
from .stage3_pinn.model import feature_vector, target_vector
from .stage3_pinn.train import predict
from .stage3_pinn.verify import classical_fallback, gate
from .stage4_settlement.coalition import CoalitionValuer, forecast_error_sd
from .stage4_settlement.governance import Action, policy, relative_spread
from .stage4_settlement.shapley import coalition_efficiency_error, shapley
from .ucproblem import make_uc_problem
from .utils import Timer


@dataclass
class LoopState:
    u_prev: np.ndarray | None = None
    up: np.ndarray | None = None
    down: np.ndarray | None = None
    theta: np.ndarray | None = None
    vm: np.ndarray | None = None
    loading: np.ndarray | None = None
    feedback: np.ndarray = field(default_factory=lambda: np.zeros(4))
    p_fixed: float = 0.0


class QPONG:
    def __init__(self, grid, cfg, model, rng: np.random.Generator, sampler=None, label="Q-PONG",
                 counts_provider=None):
        self.grid, self.cfg, self.model, self.rng = grid, cfg, model, rng
        # counts_provider(t) -> {index: count}: replaces VQE sampling (hardware replay, Sec. VI-E)
        self.counts_provider = counts_provider
        ab = cfg["ablation"]
        self.ab = ab
        self.fbwm = FBWM(cfg, adaptive=not ab["fixed_weights"])
        self.label = label
        self.sampler = sampler or (make_sampler(cfg, grid.n_enc, seed=int(rng.integers(2 ** 31)))
                                   if ab["quantum_solver"] == "vqe" else None)
        self.vqe = VQE(cfg, self.sampler, rng) if self.sampler is not None else None
        self.dt = cfg["data"]["interval_minutes"] / 60.0
        self.alpha = cfg["stage4"]["alpha_default"]

    # ------------------------------------------------------------------ helpers
    def _predict(self, X):
        return predict(self.model, X)

    def _candidates(self, prob, qubo, timer, t=None):
        """Stage II: ranked candidate commitments (encoded bits) and diagnostics."""
        diag = {}
        solver = self.ab["quantum_solver"]
        with timer("vqe"):
            if self.counts_provider is not None:
                counts = self.counts_provider(t)
                sf, ss = shot_feasible_share(prob, qubo, counts)
                diag.update(s_feas=sf, s_shots=ss)
            elif solver == "vqe":
                warm = self.cfg["stage2"]["warm_start"] and self.state.theta is not None
                E_all = qubo.energy_all() if isinstance(self.sampler, NumpySampler) else None
                res = self.vqe.run(qubo, theta0=self.state.theta if warm else None, warm=warm,
                                   energies_all=E_all)
                self.state.theta = res.theta
                counts = res.counts
                diag.update(vqe_iters=res.n_iter, vqe_energy=res.energy)
                if E_all is not None:
                    e0 = float(E_all.min())
                    diag.update(e_exact=e0, e_max=float(E_all.max()),
                                p_opt=float(res.probabilities[int(np.argmin(E_all))]))
                sf, ss = shot_feasible_share(prob, qubo, counts)
                diag.update(s_feas=sf, s_shots=ss)
                if self.cfg["stage2"].get("record_theta", False):
                    from .stage2_quantum.qubo import qubo_to_ising

                    kappa, h, J = qubo_to_ising(qubo)
                    diag.update(theta=res.theta.tolist(), ising_kappa=float(kappa), ising_h=h.tolist(),
                                ising_J=J.tolist())
            elif solver == "exact":
                Y, E, E_all = exact_enumeration(qubo, k=min(100, 1 << qubo.n))
                counts = {int((y * (1 << np.arange(qubo.n))).sum()): int(1e6 - i) for i, y in enumerate(Y)}
            elif solver == "sa":
                sa = self.ab["sa"]
                counts = {}
                for r in range(sa["restarts"]):
                    y, _ = simulated_annealing(qubo, sweeps=sa["sweeps"], restarts=1, rng=self.rng)
                    k = int((y * (1 << np.arange(qubo.n))).sum())
                    counts[k] = counts.get(k, 0) + 1
            else:
                raise ValueError(solver)
            Y, J, _ = rank_candidates(prob, qubo, counts, self.cfg)
            ok = hard_feasible(prob, Y, qubo)
            if ok.any():                       # pre-screen: drop candidates violating adequacy/min-up/down
                Y = Y[ok]
        diag["n_candidates"] = len(Y)
        if "e_exact" not in diag and qubo.n <= 20 and solver == "vqe":   # diagnostics only (untimed)
            E_all = qubo.energy_all()
            diag["e_exact"] = float(E_all.min())
            diag["e_max"] = float(E_all.max())
            if self.counts_provider is None and not isinstance(self.sampler, NumpySampler):
                from .stage2_quantum.samplers import energy_from_counts

                diag["vqe_energy_shots"] = energy_from_counts(counts, qubo)
        return Y, diag

    # ---------------------------------------------------------------- interval
    def step(self, sc, t: int) -> dict:
        grid, cfg = self.grid, self.cfg
        timer = Timer()
        rec = {"method": self.label, "t": t}
        pd, qd, ppv = sc.pd[t], sc.qd[t], sc.ppv[t]
        st = self.state

        # ---- Stage I
        with timer("bwm"):
            s = context_vector(cfg, ppv, sc.ppv_id[t], sc.price_rt[t], sc.price_ref, sc.freq[t],
                               st.vm, 1.0, st.loading)
            s = feedback_update(cfg, s, st.feedback)
            w = self.fbwm(s)
        rec.update({f"w_{k}": float(v) for k, v in zip(cfg["stage1"]["criteria"], w)})
        rec["fbwm_xi"] = self.fbwm.last_xi

        # ---- Stage II
        u_prev_enc = None if st.u_prev is None else st.u_prev[grid.encoded]
        prob = make_uc_problem(grid, cfg, w, pd, ppv, sc.pd_id[t], sc.co2_t_per_mwh, u_prev_enc,
                               st.up, st.down, st.p_fixed, st.loading)
        qubo = build_qubo(prob, cfg["stage2"]["penalties"])
        Y, diag = self._candidates(prob, qubo, timer, t)
        rec.update(diag)

        # ---- Stage III (batched PINN prediction for all candidates)
        with timer("pinn"):
            U_full = np.array([grid.full_commitment(y) for y in Y]) if len(Y) else np.zeros((0, grid.n_units))
            X = np.stack([feature_vector(grid, u, pd, qd, ppv, sc.price_rt[t], sc.price_ref) for u in U_full]) \
                if len(Y) else np.zeros((0, 1))
            Yhat = self._predict(X) if len(Y) else np.zeros((0, 1))

        price_fc = sc.price_rt[t - 1] if t > 0 else sc.price_da[t]
        n_reject, k0, fallback = 0, 0, False
        first = None
        issued = None
        action = Action.APPROVE
        settle = {}
        while True:
            chosen = None
            with timer("ver"):
                for k in range(k0, len(Y)):
                    g = gate(grid, cfg, Yhat[k], U_full[k], pd, qd, ppv, verification=self.ab["verification"])
                    if first is None:
                        first = g
                    if g.admitted:
                        chosen = k
                        break
                if chosen is None:
                    fallback = True
                    u_fb, pg_fb, vg_fb, d_fb, _ = classical_fallback(grid, pd, qd, ppv, list(U_full))
                    issued = {"u": u_fb, "pg": pg_fb, "vg": vg_fb, "disp": d_fb, "y": None}
                else:
                    issued = {"u": U_full[chosen], "pg": g.pg, "vg": g.vg, "disp": g.dispatch,
                              "y": Yhat[chosen], "repaired": g.repaired}
            if issued["disp"] is None:      # fallback solver failure
                break
            # ---- Stage IV
            with timer("set"):
                settle = self._settle(sc, t, issued["u"], price_fc)
            with timer("gov"):
                if not self.ab["hitl"]:
                    action = Action.APPROVE
                else:
                    action = policy(not fallback, settle["m_phi"], settle["s_phi"], settle["standalone"],
                                    self.alpha, cfg["stage4"]["tau_width"])
                    if fallback:
                        action = Action.FALLBACK
            if action == Action.REJECT and not fallback and n_reject < cfg["stage4"]["max_governance_rejects"]:
                n_reject += 1
                k0 = chosen + 1
                continue
            if action == Action.ADJUST:
                with timer("gov"):
                    alpha_adj = min(1.0, self.alpha + cfg["stage4"]["alpha_adjust_step"])
                    settle["alpha_used"] = alpha_adj
            break

        rec["governance"] = action.value
        rec["n_reject"] = n_reject
        rec["fallback"] = fallback
        rec["repaired"] = bool(issued and issued.get("repaired", False))

        # ---- first-pass (S1) record: top-ranked candidate, one AC check, no repair/fallback
        if first is not None:
            d1 = first.dispatch
            rec.update(s1_feasible=bool(d1.feasible), s1_nv=d1.n_v, s1_ns=d1.n_s, s1_ng=d1.n_g,
                       s1_nbal=d1.n_bal, v_phys=first.v_phys)
        else:
            rec.update(s1_feasible=False, s1_nv=0, s1_ns=0, s1_ng=0, s1_nbal=grid.n_bus)

        # ---- issued dispatch (S2)
        self._record_issued(rec, sc, t, issued)
        if issued.get("y") is not None and cfg["stage3"]["compute_reference"]:
            ref = solve_acopf(grid, pd, qd, ppv, issued["u"])
            if ref.success:
                x_ref = target_vector(grid, ref.pg, ref.qg, ref.vm, ref.va)
                rec["e_pinn"] = float(np.linalg.norm(issued["y"] - x_ref) / (np.linalg.norm(x_ref) + 1e-9))
        rec.update({k: v for k, v in settle.items() if np.isscalar(v)})
        if settle and cfg["stage4"].get("eval_coverage", True):
            # ex-post realized payoffs, for the coverage study (Fig. 16) only - not timed
            real = CoalitionValuer(grid, cfg, self._predict, issued["u"], pd, qd, ppv, sc.price_rt[t],
                                   sc.price_ref, 0.0, 0.0, 0.0, self.dt)
            s4 = cfg["stage4"]
            m_real, _, _ = shapley(grid.n_enc, real, s4["shapley_exact_max_n"], s4["n_permutations"],
                                   np.random.default_rng(t))
            rec["alloc_m"] = settle["m_phi"].tolist()
            rec["alloc_s"] = settle["s_phi"].tolist()
            rec["alloc_real"] = m_real.tolist()
            rec["alloc_standalone"] = settle["standalone"].tolist()

        # ---- feedback e_t (eq. 21) and state update
        d = first.dispatch if first is not None else None
        e_t = np.zeros(4)
        if d is not None and (not d.feasible or action == Action.REJECT):
            e_t = np.array([min(d.max_vdev / cfg["stage1"]["vdev_max"], 1.0),
                            max(d.max_loading - 1.0, 0.0), float(d.n_bal > 0),
                            float(action == Action.REJECT)])
        st.feedback = e_t
        self._update_state(issued)
        rec["times"] = dict(timer.t)
        rec["t_e2e"] = timer.total()
        return rec

    # ---------------------------------------------------------------- settlement
    def _settle(self, sc, t, u, price_fc) -> dict:
        grid, cfg = self.grid, self.cfg
        W = cfg["stage4"]["spread_window"]
        lo = max(0, t - W + 1)
        fuzzy = self.ab["fuzzy_settlement"]
        if fuzzy:
            sig_res = forecast_error_sd(sc.ppv[lo:t + 1].sum(1), sc.ppv_id[lo:t + 1].sum(1)) if t > 1 else 0.0
            sig_load = forecast_error_sd(sc.pd[lo:t + 1].sum(1), sc.pd_id[lo:t + 1].sum(1)) if t > 1 else 0.0
            pr = sc.price_rt[lo:t + 1]
            sig_price = float(np.std(np.diff(pr))) if len(pr) > 2 else 0.0
        else:
            sig_res = sig_load = sig_price = 0.0
        valuer = CoalitionValuer(grid, cfg, self._predict, u, sc.pd_id[t], sc.qd[t], sc.ppv_id[t],
                                 price_fc, sc.price_ref, sig_res, sig_load, sig_price, self.dt)
        n = grid.n_enc
        s4 = cfg["stage4"]
        m_phi, s_phi, info = shapley(n, valuer, s4["shapley_exact_max_n"], s4["n_permutations"], self.rng)
        m_grand, _ = valuer(np.array([(1 << n) - 1]))
        singles, _ = valuer(np.array([1 << i for i in range(n)]))
        out = {"m_phi": m_phi, "s_phi": s_phi, "standalone": singles,
               "e_coal": coalition_efficiency_error(m_phi, float(m_grand[0])),
               "n_coalitions": info["n_coalitions"],
               "sigma_rel_mean": float(np.mean(relative_spread(m_phi, s_phi))),
               "alpha_used": self.alpha}
        return out

    def _record_issued(self, rec, sc, t, issued):
        grid = self.grid
        pd, qd, ppv = sc.pd[t], sc.qd[t], sc.ppv[t]
        d = issued["disp"]
        u = issued["u"]
        if d is None or not d.converged:
            rec.update(s2_feasible=False, s2_nv=0, s2_ns=0, s2_ng=0, s2_nbal=grid.n_bus,
                       cost=np.nan, e_ac=np.nan, fallback_failure=d is None)
            return
        cost = grid.gen_cost(d.pg, u, self.dt) + grid.startup_cost(u, self.state.u_prev)
        rec.update(s2_feasible=bool(d.feasible), s2_nv=d.n_v, s2_ns=d.n_s, s2_ng=d.n_g, s2_nbal=d.n_bal,
                   cost=cost, energy_mwh=float((pd.sum() - ppv.sum()) * self.dt),
                   carbon_t=float(d.pg.sum() * self.dt * sc.co2_t_per_mwh),
                   e_ac=ac_mismatch(grid, d.vm, d.va, d.pg, d.qg, pd, qd, ppv),
                   n_committed=int(u[grid.encoded].sum()), fallback_failure=False)
        rec["u_enc"] = "".join(str(int(x)) for x in u[grid.encoded])

    def _update_state(self, issued):
        grid, st = self.grid, self.state
        u = issued["u"]
        ue = u[grid.encoded]
        if st.u_prev is None:
            st.up = np.full(grid.n_enc, 99)
            st.down = np.full(grid.n_enc, 99)
        else:
            prev = st.u_prev[grid.encoded]
            st.up = np.where(ue > 0.5, np.where(prev > 0.5, st.up + 1, 1), 0)
            st.down = np.where(ue < 0.5, np.where(prev < 0.5, st.down + 1, 1), 0)
        st.u_prev = u.copy()
        d = issued["disp"]
        if d is not None and d.converged:
            st.vm = d.vm
            from .grid.acpf import branch_loading

            st.loading = branch_loading(grid, d.vm, d.va)
            if len(grid.fixed):
                st.p_fixed = float(d.pg[grid.fixed].sum())

    # -------------------------------------------------------------------- day
    def run_day(self, sc, intervals=None) -> list[dict]:
        self.state = LoopState()
        if len(self.grid.fixed):
            # initial estimate of the fixed units' output: base-case OPF scaled to the first interval
            r = solve_acopf(self.grid, sc.pd[0], sc.qd[0], sc.ppv[0], np.ones(self.grid.n_units))
            self.state.p_fixed = float(r.pg[self.grid.fixed].sum()) if r.success else 0.0
        out = []
        for t in (intervals if intervals is not None else range(sc.T)):
            rec = self.step(sc, t)
            rec.update(system=sc.system, day=sc.day, penetration=sc.penetration, stress=sc.stress,
                       data_source=sc.data_source)
            out.append(rec)
        return out
