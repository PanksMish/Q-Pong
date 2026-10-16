"""Physics-informed operating-point predictor f_phi (eq. 15) and its loss (eqs. 36-40).

Input q_t^(k):  commitment u (all units), bus active/reactive load, bus PV,
                and the normalized real-time price.
Output x_hat:   [P_g, Q_g (per unit), V (per bus), theta (per bus)], in pu/rad.

Outputs are de-normalized with training statistics; P_g and Q_g are
multiplied by u so that decommitted units inject nothing. Limits are NOT
enforced by the architecture: they enter the loss as penalties (37)-(39),
and feasibility is established only by the independent gate.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GridTensors:
    """Network data as torch tensors (pu) for the physics terms."""

    def __init__(self, grid, device="cpu", dtype=torch.float32):
        U = grid.units
        bm = grid.base_mva
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=dtype, device=device)  # noqa: E731
        self.G, self.B = t(grid.G), t(grid.B)
        self.C = t(grid.unit_bus_matrix())
        self.pmin, self.pmax = t(U.pmin / bm), t(U.pmax / bm)
        self.qmin, self.qmax = t(U.qmin / bm), t(U.qmax / bm)
        self.vmin, self.vmax = t(grid.vmin), t(grid.vmax)
        Yf, Yt = grid.Yf.toarray(), grid.Yt.toarray()
        self.Yf_r, self.Yf_i = t(Yf.real), t(Yf.imag)
        self.Yt_r, self.Yt_i = t(Yt.real), t(Yt.imag)
        self.f, self.tb = torch.as_tensor(grid.f_bus, device=device), torch.as_tensor(grid.t_bus, device=device)
        smax = np.where(np.isfinite(grid.s_max), grid.s_max, 1e6)
        self.smax = t(smax)
        self.slack = grid.slack_bus
        self.nb, self.ng = grid.n_bus, grid.n_units


def feature_vector(grid, u, pd_mw, qd_mvar, ppv_mw, price, price_ref):
    bm = grid.base_mva
    return np.concatenate([np.asarray(u, float), pd_mw / bm, qd_mvar / bm, ppv_mw / bm,
                           [(price - price_ref) / (abs(price_ref) + 1e-9)]]).astype(np.float32)


def target_vector(grid, pg, qg, vm, va):
    bm = grid.base_mva
    return np.concatenate([pg / bm, qg / bm, vm, va]).astype(np.float32)


class PINN(nn.Module):
    def __init__(self, n_in, gt: GridTensors, hidden=(128, 128, 128, 128), act="tanh"):
        super().__init__()
        A = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[act]
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), A()]
            d = h
        layers.append(nn.Linear(d, 2 * gt.ng + 2 * gt.nb))
        self.net = nn.Sequential(*layers)
        self.gt = gt
        self.register_buffer("x_mu", torch.zeros(n_in))
        self.register_buffer("x_sd", torch.ones(n_in))
        self.register_buffer("y_mu", torch.zeros(2 * gt.ng + 2 * gt.nb))
        self.register_buffer("y_sd", torch.ones(2 * gt.ng + 2 * gt.nb))

    def set_normalization(self, X, Y):
        self.x_mu.copy_(torch.as_tensor(X.mean(0)))
        self.x_sd.copy_(torch.as_tensor(X.std(0) + 1e-6))
        self.y_mu.copy_(torch.as_tensor(Y.mean(0)))
        self.y_sd.copy_(torch.as_tensor(Y.std(0) + 1e-6))

    def forward(self, x):
        gt = self.gt
        y = self.net((x - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu
        u = x[:, : gt.ng]
        pg = y[:, : gt.ng] * u
        qg = y[:, gt.ng: 2 * gt.ng] * u
        vm = y[:, 2 * gt.ng: 2 * gt.ng + gt.nb]
        va = y[:, 2 * gt.ng + gt.nb:]
        va = va - va[:, gt.slack: gt.slack + 1]          # reference angle = 0
        return torch.cat([pg, qg, vm, va], dim=1)

    def split(self, y):
        gt = self.gt
        return (y[:, : gt.ng], y[:, gt.ng: 2 * gt.ng], y[:, 2 * gt.ng: 2 * gt.ng + gt.nb],
                y[:, 2 * gt.ng + gt.nb:])


def physics_terms(model: PINN, x, y):
    """Return dict of the loss terms L_AC, L_V, L_S, L_G (eqs. 36-39) per batch."""
    gt = model.gt
    pg, qg, vm, va = model.split(y)
    u = x[:, : gt.ng]
    nb = gt.nb
    pd = x[:, gt.ng: gt.ng + nb]
    qd = x[:, gt.ng + nb: gt.ng + 2 * nb]
    ppv = x[:, gt.ng + 2 * nb: gt.ng + 3 * nb]
    pinj = pg @ gt.C.T - pd + ppv
    qinj = qg @ gt.C.T - qd
    th = va[:, :, None] - va[:, None, :]
    cos, sin = torch.cos(th), torch.sin(th)
    vv = vm[:, :, None] * vm[:, None, :]
    pcalc = (vv * (gt.G * cos + gt.B * sin)).sum(-1)
    qcalc = (vv * (gt.G * sin - gt.B * cos)).sum(-1)
    p_scale = pd.abs().mean() + 1e-3
    q_scale = qd.abs().mean() + 1e-3
    L_ac = (((pinj - pcalc) / p_scale) ** 2 + ((qinj - qcalc) / q_scale) ** 2).mean()
    relu = torch.relu
    L_v = (relu(gt.vmin - vm) ** 2 + relu(vm - gt.vmax) ** 2).mean()
    vr, vi = vm * torch.cos(va), vm * torch.sin(va)
    ifr = vr @ gt.Yf_r.T - vi @ gt.Yf_i.T
    ifi = vr @ gt.Yf_i.T + vi @ gt.Yf_r.T
    itr = vr @ gt.Yt_r.T - vi @ gt.Yt_i.T
    iti = vr @ gt.Yt_i.T + vi @ gt.Yt_r.T
    sf = torch.sqrt((vm[:, gt.f] ** 2) * (ifr ** 2 + ifi ** 2) + 1e-12)
    st = torch.sqrt((vm[:, gt.tb] ** 2) * (itr ** 2 + iti ** 2) + 1e-12)
    L_s = (relu(sf - gt.smax) ** 2 + relu(st - gt.smax) ** 2).mean()
    L_g = (relu(pg - gt.pmax * u) ** 2 + relu(gt.pmin * u - pg) ** 2
           + relu(qg - gt.qmax * u) ** 2 + relu(gt.qmin * u - qg) ** 2).sum(1).mean()
    return {"AC": L_ac, "V": L_v, "S": L_s, "G": L_g}


def pinn_loss(model, x, y_pred, y_ref, lambdas, L0, physics=True):
    """Eq. (40): sum_k lambda_k L_k / (L_k^(0) + eps)."""
    terms = physics_terms(model, x, y_pred) if physics else {}
    terms["D"] = ((y_pred - y_ref) ** 2).sum(1).mean()
    total = 0.0
    for k, v in terms.items():
        total = total + lambdas.get(k, 1.0) * v / (L0.get(k, 1.0) + 1e-8)
    return total, {k: float(v.detach()) for k, v in terms.items()}
