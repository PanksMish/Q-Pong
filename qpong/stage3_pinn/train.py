"""Offline PINN training (Section V-C): Adam 1e-3, LR halved after 100 stagnant
epochs, early stop after 200 epochs without validation improvement, <= 2000 epochs.
Each loss term is normalized by its value at initialization (eq. 40).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..utils import get_logger
from .model import PINN, GridTensors, physics_terms, pinn_loss

log = get_logger("qpong.pinn")


def train_pinn(grid, cfg, Xtr, Ytr, Xva, Yva, physics=True, seed=0, device="cpu", hidden=None,
               verbose=True):
    s3 = cfg["stage3"]
    torch.manual_seed(seed)
    gt = GridTensors(grid, device)
    model = PINN(Xtr.shape[1], gt, hidden or s3["hidden"], s3["activation"]).to(device)
    model.set_normalization(Xtr, Ytr)
    Xtr_t, Ytr_t = torch.as_tensor(Xtr, device=device), torch.as_tensor(Ytr, device=device)
    Xva_t, Yva_t = torch.as_tensor(Xva, device=device), torch.as_tensor(Yva, device=device)
    lambdas = s3["lambdas"]
    with torch.no_grad():
        idx = torch.randperm(len(Xtr_t))[: min(1024, len(Xtr_t))]
        y0 = model(Xtr_t[idx])
        L0 = {k: float(v) for k, v in physics_terms(model, Xtr_t[idx], y0).items()} if physics else {}
        L0["D"] = float(((y0 - Ytr_t[idx]) ** 2).sum(1).mean())
        L0 = {k: max(v, 1e-6) for k, v in L0.items()}
    opt = torch.optim.Adam(model.parameters(), lr=s3["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=s3["lr_patience"])
    best, best_state, stall = np.inf, None, 0
    bs = s3["batch_size"]
    t0 = time.perf_counter()
    for ep in range(s3["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss, _ = pinn_loss(model, Xtr_t[b], model(Xtr_t[b]), Ytr_t[b], lambdas, L0, physics)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl, parts = pinn_loss(model, Xva_t, model(Xva_t), Yva_t, lambdas, L0, physics)
            vl = float(vl)
        sched.step(vl)
        if vl < best - 1e-7:
            best, stall = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
        if verbose and ep % 50 == 0:
            log.info("epoch %d val %.4e %s", ep, vl, {k: f"{v:.2e}" for k, v in parts.items()})
        if stall >= s3["early_stop_patience"]:
            break
    model.load_state_dict(best_state)
    info = {"epochs": ep + 1, "best_val": best, "train_time_s": time.perf_counter() - t0, "L0": L0,
            "physics": physics}
    return model, info


def save_model(model, info, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "n_in": model.x_mu.numel(), "info": info,
                "hidden": [m.out_features for m in model.net if isinstance(m, torch.nn.Linear)][:-1]}, path)


def load_model(grid, path, device="cpu", act="tanh"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = PINN(ck["n_in"], GridTensors(grid, device), ck["hidden"], act).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["info"]


@torch.no_grad()
def predict(model, X: np.ndarray) -> np.ndarray:
    return model(torch.as_tensor(np.atleast_2d(X), dtype=torch.float32)).cpu().numpy()
