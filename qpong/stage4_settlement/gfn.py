"""Gaussian fuzzy numbers (Section III-C): membership, alpha-cuts (3), addition (4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GFN:
    m: float
    s: float

    def membership(self, x):
        return np.exp(-((np.asarray(x) - self.m) ** 2) / (2 * self.s ** 2)) if self.s > 0 else (np.asarray(x) == self.m).astype(float)

    def alpha_cut(self, alpha: float):
        return alpha_cut(self.m, self.s, alpha)

    def __add__(self, o: "GFN") -> "GFN":
        return GFN(self.m + o.m, float(np.hypot(self.s, o.s)))

    def __sub__(self, o: "GFN") -> "GFN":
        return GFN(self.m - o.m, float(np.hypot(self.s, o.s)))

    def scale(self, w: float) -> "GFN":
        return GFN(w * self.m, abs(w) * self.s)


def half_width(s, alpha):
    alpha = np.clip(alpha, 1e-12, 1.0)
    return np.asarray(s) * np.sqrt(-2.0 * np.log(alpha))


def alpha_cut(m, s, alpha):
    h = half_width(s, alpha)
    return np.asarray(m) - h, np.asarray(m) + h


def width(s, alpha):
    """W(alpha) = 2 sigma sqrt(-2 ln alpha)  (eq. 19)."""
    return 2.0 * half_width(s, alpha)


def gaussian_coverage(alpha):
    """1 - 2 Phi(-sqrt(-2 ln alpha)): coverage if sigma were a calibrated std."""
    from scipy.stats import norm

    return 1 - 2 * norm.cdf(-np.sqrt(-2 * np.log(np.clip(alpha, 1e-12, 1))))
