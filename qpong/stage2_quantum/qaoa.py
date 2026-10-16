"""QAOA on a diagonal (QUBO) cost, exact statevector simulation (used by the QDUC baseline)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class QAOA:
    def __init__(self, energies_all: np.ndarray, n: int, p: int = 2):
        self.E = energies_all
        self.n, self.p = n, p
        self.scale = max(np.ptp(energies_all), 1e-12)

    def state(self, params):
        n, E = self.n, self.E / self.scale
        psi = np.full(1 << n, 1 / np.sqrt(1 << n), dtype=complex)
        gam, bet = params[: self.p], params[self.p:]
        for g, b in zip(gam, bet):
            psi = psi * np.exp(-1j * g * E)
            c, s = np.cos(b), -1j * np.sin(b)
            for q in range(n):
                v = psi.reshape(1 << (n - q - 1), 2, 1 << q)
                a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
                v[:, 0, :] = c * a0 + s * a1
                v[:, 1, :] = s * a0 + c * a1
                psi = v.reshape(-1)
        return psi

    def expectation(self, params):
        pr = np.abs(self.state(params)) ** 2
        return float(pr @ self.E)

    def optimize(self, maxiter=100, rng=None):
        rng = rng or np.random.default_rng()
        x0 = np.r_[rng.uniform(0, np.pi, self.p), rng.uniform(0, np.pi / 2, self.p)]
        res = minimize(self.expectation, x0, method="COBYLA", options={"maxiter": maxiter})
        pr = np.abs(self.state(res.x)) ** 2
        return res.x, pr
