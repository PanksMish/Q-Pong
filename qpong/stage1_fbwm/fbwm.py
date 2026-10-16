"""Context-adaptive fuzzy Best-Worst Method (Stage I, Algorithm 1, eqs. 32-33).

Preferences are triangular fuzzy numbers (l, m, r) with graded-mean
defuzzification D(a) = (l + m + r) / 3. Each baseline preference is shifted
by the context term gamma_ij . s_t and clipped to [a_min, a_max]; the
adapted midpoint a_bar enters the min-max program (33):

    min xi  s.t. |w_B / w_j - a_Bj| <= xi,  |w_j / w_W - a_jW| <= xi,
                 sum_j w_j = 1, w_j >= 0.

The program is solved with SLSQP, warm-started from the linear BWM solution
(Rezaei 2016), which is solved exactly as an LP.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize


def tfn_shift(a0, shift, a_min, a_max):
    return np.clip(np.asarray(a0, float) + shift, a_min, a_max)


def defuzzify(a) -> float:
    return float(np.mean(a))


def _linear_bwm(aB, aW, b, w):
    K = len(aB)
    # variables: w_1..w_K, xi ;  |w_b - aB_j w_j| <= xi ; |w_j - aW_j w_w| <= xi
    A, ub = [], []
    for j in range(K):
        row = np.zeros(K + 1)
        row[b] += 1
        row[j] -= aB[j]
        for s in (1, -1):
            r = s * row.copy()
            r[-1] = -1
            A.append(r)
            ub.append(0.0)
        row = np.zeros(K + 1)
        row[j] += 1
        row[w] -= aW[j]
        for s in (1, -1):
            r = s * row.copy()
            r[-1] = -1
            A.append(r)
            ub.append(0.0)
    Aeq = np.r_[np.ones(K), 0.0][None, :]
    c = np.r_[np.zeros(K), 1.0]
    res = linprog(c, A_ub=np.array(A), b_ub=ub, A_eq=Aeq, b_eq=[1.0],
                  bounds=[(0, 1)] * K + [(0, None)], method="highs")
    return res.x[:K] if res.success else np.full(K, 1.0 / K)


def solve_bwm(aB: np.ndarray, aW: np.ndarray, b: int, w: int):
    """Solve program (33). Returns (weights, xi*)."""
    K = len(aB)
    x0 = np.clip(_linear_bwm(aB, aW, b, w), 1e-4, 1)
    x0 = x0 / x0.sum()
    eps = 1e-6

    def obj(z):
        return z[-1]

    cons = [{"type": "eq", "fun": lambda z: np.sum(z[:K]) - 1.0}]
    for j in range(K):
        cons += [
            {"type": "ineq", "fun": lambda z, j=j: z[-1] - (z[b] / max(z[j], eps) - aB[j])},
            {"type": "ineq", "fun": lambda z, j=j: z[-1] + (z[b] / max(z[j], eps) - aB[j])},
            {"type": "ineq", "fun": lambda z, j=j: z[-1] - (z[j] / max(z[w], eps) - aW[j])},
            {"type": "ineq", "fun": lambda z, j=j: z[-1] + (z[j] / max(z[w], eps) - aW[j])},
        ]
    xi0 = max(max(abs(x0[b] / x0[j] - aB[j]), abs(x0[j] / x0[w] - aW[j])) for j in range(K))
    res = minimize(obj, np.r_[x0, xi0], method="SLSQP", constraints=cons,
                   bounds=[(eps, 1)] * K + [(0, None)], options={"maxiter": 200, "ftol": 1e-10})
    z = res.x if res.success else np.r_[x0, xi0]
    wts = np.clip(z[:K], 0, None)
    return wts / wts.sum(), float(z[-1])


class FBWM:
    """Phi_FBWM: context s_t -> weights w_t on the simplex."""

    def __init__(self, cfg: dict, adaptive: bool = True):
        c = cfg["stage1"]
        self.crit = list(c["criteria"])
        self.b = self.crit.index(c["best"])
        self.w = self.crit.index(c["worst"])
        self.BO0 = np.array([c["best_to_others"][k] for k in self.crit], float)      # (K, 3)
        self.OW0 = np.array([c["others_to_worst"][k] for k in self.crit], float)
        self.gBO = np.array([c["gamma_best_to_others"][k] for k in self.crit], float)  # (K, 4)
        self.gOW = np.array([c["gamma_others_to_worst"][k] for k in self.crit], float)
        self.amin, self.amax = c["a_min"], c["a_max"]
        self.adaptive = adaptive
        self.last_xi = 0.0

    def adapted_preferences(self, s: np.ndarray):
        s = np.zeros(4) if not self.adaptive else np.asarray(s, float)
        BO = np.array([tfn_shift(self.BO0[j], self.gBO[j] @ s, self.amin, self.amax) for j in range(len(self.crit))])
        OW = np.array([tfn_shift(self.OW0[j], self.gOW[j] @ s, self.amin, self.amax) for j in range(len(self.crit))])
        BO[self.b] = 1.0
        OW[self.w] = 1.0
        return BO, OW

    def __call__(self, s: np.ndarray) -> np.ndarray:
        BO, OW = self.adapted_preferences(s)
        aB = BO[:, 1]                     # adapted midpoint a_bar (eq. 32)
        aW = OW[:, 1]
        wts, xi = solve_bwm(aB, aW, self.b, self.w)
        self.last_xi = xi
        return wts

    def defuzzified_weights(self, s: np.ndarray) -> np.ndarray:
        """Variant using graded-mean defuzzification instead of the midpoint."""
        BO, OW = self.adapted_preferences(s)
        aB = np.array([defuzzify(a) for a in BO])
        aW = np.array([defuzzify(a) for a in OW])
        return solve_bwm(aB, aW, self.b, self.w)[0]
