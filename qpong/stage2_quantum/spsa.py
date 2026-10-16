"""Simultaneous perturbation stochastic approximation (Spall 1992).

Two objective evaluations per iteration, independent of the number of
parameters. Gains a_k = a / (k + 1 + A)^alpha, c_k = c / (k + 1)^gamma.
"""
from __future__ import annotations

import numpy as np


def calibrate(f, theta0, c=0.1, target_step=2 * np.pi / 10, alpha=0.602, A=30, n_samples=25, rng=None):
    """Learning-rate calibration (as in Qiskit's SPSA.calibrate): choose ``a`` so that
    the first update has magnitude ~ ``target_step``. Costs 2 * n_samples evaluations."""
    rng = rng or np.random.default_rng()
    mags = []
    for _ in range(n_samples):
        delta = rng.choice([-1.0, 1.0], size=np.shape(theta0))
        mags.append(abs(f(theta0 + c * delta) - f(theta0 - c * delta)) / (2 * c))
    g = max(float(np.mean(mags)), 1e-10)
    return target_step * (A + 1) ** alpha / g, 2 * n_samples


def spsa_minimize(f, theta0, maxiter=300, a=0.2, c=0.1, alpha=0.602, gamma=0.101, A=30,
                  rng: np.random.Generator | None = None, callback=None, tol=None, patience=30):
    rng = rng or np.random.default_rng()
    theta = np.array(theta0, float)
    history = []
    best = (np.inf, theta.copy())
    stall = 0
    n_eval = 0
    for k in range(maxiter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        fp = f(theta + ck * delta)
        fm = f(theta - ck * delta)
        n_eval += 2
        g = (fp - fm) / (2 * ck) * delta
        theta = theta - ak * g
        fk = 0.5 * (fp + fm)
        history.append(fk)
        if callback:
            callback(k, theta, fk)
        if fk < best[0] - (tol or 0.0):
            best = (fk, theta.copy())
            stall = 0
        else:
            stall += 1
        if tol is not None and stall >= patience:
            break
    return {"x": theta, "fun": history[-1] if history else np.nan, "nit": len(history),
            "nfev": n_eval, "history": np.array(history), "best_x": best[1], "best_fun": best[0]}
