"""alpha-cut governance (eqs. 19-20) and the automated operator policy H.

H returns
  FALLBACK if no candidate passed verification;
  REJECT   if the lower alpha-cut bound of any participant's allocation is
           below its stand-alone value m_{i};
  ADJUST   if max_i W_i(alpha) / |m_phi_i| > tau (tau = 0.15);
  APPROVE  otherwise.
In deployment H is a human operator; here it is this automated policy, and
the governance time covers only the interval computation and the policy.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from .gfn import alpha_cut, width


class Action(str, Enum):
    APPROVE = "APPROVE"
    ADJUST = "ADJUST"
    REJECT = "REJECT"
    FALLBACK = "FALLBACK"


def policy(verified: bool, m_phi, s_phi, standalone, alpha, tau=0.15) -> Action:
    if not verified:
        return Action.FALLBACK
    lo, _ = alpha_cut(m_phi, s_phi, alpha)
    if np.any(lo < standalone - 1e-9):
        return Action.REJECT
    rel = width(s_phi, alpha) / np.maximum(np.abs(m_phi), 1e-9)
    if np.max(rel) > tau:
        return Action.ADJUST
    return Action.APPROVE


def relative_spread(m_phi, s_phi):
    return np.asarray(s_phi) / np.maximum(np.abs(m_phi), 1e-9)
