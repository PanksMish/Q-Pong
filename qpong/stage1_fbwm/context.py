"""Context indicators (eqs. 7, 31) and the closed-loop feedback update (eq. 21)."""
from __future__ import annotations

import numpy as np

EPS = 1e-9
LOWER = np.array([0.0, 0.0, -1.0, 0.0])
UPPER = np.array([1.0, 1.0, 1.0, 1.0])


def project(s: np.ndarray) -> np.ndarray:
    """Pi_S: projection onto the admissible context domain."""
    return np.clip(s, LOWER, UPPER)


def context_vector(cfg: dict, ppv, ppv_hat, price, price_ref, freq, vm_meas=None, vref=1.0,
                   loading_meas=None) -> np.ndarray:
    """s_t = [nu, rho, pi, delta_f]."""
    c = cfg["stage1"]
    nu = np.linalg.norm(ppv - ppv_hat) / (np.linalg.norm(ppv_hat) + EPS)
    rho_v = 0.0 if vm_meas is None else float(np.max(np.abs(vm_meas - vref)) / c["vdev_max"])
    rho_s = 0.0 if loading_meas is None or len(loading_meas) == 0 else float(np.max(loading_meas))
    rho = max(rho_v, rho_s)
    pi = (price - price_ref) / (abs(price_ref) + EPS)
    df = abs(freq - c["freq_nominal"]) / c["freq_dev_max"]
    return project(np.array([nu, rho, pi, df], dtype=float))


def feedback_update(cfg: dict, s_next: np.ndarray, e_t: np.ndarray) -> np.ndarray:
    """Eq. (21): s_{t+1} = Pi_S(s_{t+1} + Gamma_e e_t).

    e_t = [voltage deviation, thermal overload, power-balance, governance flag]
    (normalized); Gamma_e is diagonal with ``stage1.feedback_gain``.
    """
    gain = np.asarray(cfg["stage1"]["feedback_gain"], float)
    return project(s_next + gain * np.asarray(e_t, float))
