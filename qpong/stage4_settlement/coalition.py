"""Coalition values v~(S) = (m_S, sigma_S) (eqs. 41-42).

Participants are the encoded units (generators / thermal zones). For a
coalition S the commitment is u_S = u* restricted to S (classically fixed
units stay on); ONE PINN prediction gives the coalition operating point
(batched over coalitions, cached by bitmask). Then

  m_S = R_S - C_gen_S - C_network_S - C_market_S
      R_S          = price * sum_{i in S} P_i * dt
      C_gen_S      = sum_{i in S} (a P^2 + b P + c) dt
      C_network_S  = network tariff * sum_{i in S} P_i * dt
      C_market_S   = imbalance factor * price * max(0, kappa_S * NL - sum_{i in S} P_i) * dt
                     (kappa_S = capacity share of S, NL = net load served by encoded units)
  sigma_S = sqrt( w_r sig_RES^2 + w_d sig_load^2 + w_p sig_price^2 ),
      sig_RES_S   = price * sd(RES forecast error) * kappa_S * dt
      sig_load_S  = price * sd(load forecast error) * kappa_S * dt
      sig_price_S = sd(price forecast error) * sum_{i in S} P_i * dt
The empty coalition earns 0 and incurs 0 (m_empty = 0, Proposition 2).
The terms of (41)-(42) are a MODELING CHOICE where the paper is not explicit.
"""
from __future__ import annotations

import numpy as np

from ..stage3_pinn.model import feature_vector


def masks_to_bits(masks, n):
    masks = np.asarray(masks, dtype=np.int64)
    return ((masks[:, None] >> np.arange(n)) & 1).astype(float)


class CoalitionValuer:
    def __init__(self, grid, cfg, predict_fn, u_star, pd, qd, ppv, price, price_ref, sig_res, sig_load,
                 sig_price, dt_h=0.25):
        self.grid, self.cfg = grid, cfg
        self.predict = predict_fn
        self.u_star = np.asarray(u_star, float)
        self.pd, self.qd, self.ppv = pd, qd, ppv
        self.price, self.price_ref = price, price_ref
        self.sig = (sig_res, sig_load, sig_price)
        self.dt = dt_h
        self.n = grid.n_enc
        U = grid.units
        enc = grid.encoded
        self.pmax_enc = U.pmax.to_numpy()[enc]
        self.cap_share_den = self.pmax_enc.sum()
        self.cache: dict[int, tuple[float, float]] = {}
        fixed_out = 0.0
        self.netload_enc = max(pd.sum() - ppv.sum() - fixed_out, 0.0)
        self.batch = cfg["stage4"]["pinn_batch"]
        self.n_pinn_calls = 0

    def _values(self, masks):
        grid, n = self.grid, self.n
        bits = masks_to_bits(masks, n)
        U_all = np.tile(grid.full_commitment(np.ones(n)), (len(masks), 1))
        U_all[:, grid.encoded] = bits * self.u_star[grid.encoded]
        X = np.stack([feature_vector(grid, u, self.pd, self.qd, self.ppv, self.price, self.price_ref) for u in U_all])
        Y = np.concatenate([self.predict(X[i:i + self.batch]) for i in range(0, len(X), self.batch)])
        self.n_pinn_calls += len(X)
        pg = Y[:, : grid.n_units] * grid.base_mva
        pg_enc = np.clip(pg[:, grid.encoded], 0, None) * bits
        Un = grid.units
        a, b, c = (Un[k].to_numpy()[grid.encoded] for k in ("a", "b", "c"))
        on = (bits * self.u_star[grid.encoded]) > 0.5
        dt = self.dt
        o = self.cfg["objectives"]
        energy = pg_enc.sum(1) * dt
        R = self.price * energy
        Cg = ((a * pg_enc ** 2 + b * pg_enc + c * on).sum(1)) * dt
        Cn = o["network_cost_per_mwh"] * energy
        kappa = (bits * self.pmax_enc).sum(1) / self.cap_share_den
        short = np.clip(kappa * self.netload_enc - pg_enc.sum(1), 0, None)
        Cm = o["imbalance_price_factor"] * abs(self.price) * short * dt
        m = R - Cg - Cn - Cm
        w = self.cfg["stage4"]["spread_weights"]
        s_res = abs(self.price) * self.sig[0] * kappa * dt
        s_load = abs(self.price) * self.sig[1] * kappa * dt
        s_price = self.sig[2] * energy
        s = np.sqrt(w["res"] * s_res ** 2 + w["load"] * s_load ** 2 + w["price"] * s_price ** 2)
        empty = np.asarray(masks) == 0
        m[empty] = 0.0
        s[empty] = 0.0
        return m, s

    def __call__(self, masks):
        masks = np.asarray(masks, dtype=np.int64)
        todo = np.array([k for k in masks.tolist() if k not in self.cache], dtype=np.int64)
        if len(todo):
            m, s = self._values(todo)
            for k, mi, si in zip(todo.tolist(), m, s):
                self.cache[k] = (float(mi), float(si))
        out = np.array([self.cache[k] for k in masks.tolist()])
        return out[:, 0], out[:, 1]


def forecast_error_sd(realized: np.ndarray, forecast: np.ndarray) -> float:
    e = np.asarray(realized, float) - np.asarray(forecast, float)
    return float(np.std(e)) if len(e) > 1 else 0.0
