"""VQE candidate generator (Stage II, Algorithm 2).

The QUBO energy is estimated from S_shots measurements of the ansatz state;
SPSA updates the parameters. Operational runs are warm-started from the
previous interval's optimum and stop on convergence; the standalone sweeps
(Table V/VI) start from random parameters and run all ``maxiter`` iterations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .ansatz import n_params
from .samplers import NumpySampler, energy_from_counts
from .spsa import calibrate, spsa_minimize


@dataclass
class VQEResult:
    theta: np.ndarray = field(repr=False)
    energy: float                     # final (exact if available, else shot) energy estimate
    n_iter: int
    n_eval: int
    time_s: float
    counts: dict = field(repr=False, default_factory=dict)
    probabilities: np.ndarray | None = field(repr=False, default=None)
    history: np.ndarray | None = field(repr=False, default=None)


class VQE:
    def __init__(self, cfg: dict, sampler, rng: np.random.Generator):
        self.cfg = cfg
        self.s2 = cfg["stage2"]
        self.sampler = sampler
        self.rng = rng

    def run(self, qubo, theta0=None, warm: bool = False, energies_all=None) -> VQEResult:
        s2 = self.s2
        n = qubo.n
        K = n_params(n, s2["reps"])
        t0 = time.perf_counter()
        if theta0 is None or len(theta0) != K:
            theta0 = self.rng.uniform(0, 2 * np.pi, K)
        shots = s2["shots"]
        if isinstance(self.sampler, NumpySampler):
            if energies_all is None:
                energies_all = qubo.energy_all()
            scale = max(np.ptp(energies_all), 1e-9)

            def f(th):
                return self.sampler.expectation(th, shots, self.rng, energies_all) / scale
        else:
            scale = 1.0

            def f(th):
                return energy_from_counts(self.sampler.sample(th, shots, self.rng), qubo)

        sp = s2["spsa"]
        a, n_cal = sp["a"], 0
        if sp.get("calibrate", True):
            target = sp.get("target_step", 2 * np.pi / 10) * (sp.get("warm_step_factor", 0.2) if warm else 1.0)
            a, n_cal = calibrate(f, theta0, c=sp["c"], target_step=target, alpha=sp["alpha"], A=sp["A"],
                                 n_samples=sp.get("calibration_samples", 25), rng=self.rng)
        kwargs = dict(a=a, c=sp["c"], alpha=sp["alpha"], gamma=sp["gamma"], A=sp["A"], rng=self.rng)
        if warm:
            res = spsa_minimize(f, theta0, maxiter=s2.get("warm_maxiter", s2["maxiter"]),
                                tol=s2.get("spsa_tol", 1e-4), patience=s2.get("spsa_patience", 30), **kwargs)
        else:
            res = spsa_minimize(f, theta0, maxiter=s2["maxiter"], **kwargs)
        theta = res["x"]
        counts = self.sampler.sample(theta, shots, self.rng)
        probs = None
        if isinstance(self.sampler, NumpySampler):
            probs = self.sampler.probabilities(theta)
            energy = float(probs @ energies_all)
        else:
            energy = energy_from_counts(counts, qubo)
        return VQEResult(theta=theta, energy=energy, n_iter=res["nit"], n_eval=res["nfev"] + n_cal,
                         time_s=time.perf_counter() - t0, counts=counts, probabilities=probs,
                         history=res["history"] * scale)
