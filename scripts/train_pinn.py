#!/usr/bin/env python
"""Generate AC-OPF training data and train the PINN variants (offline; not in the latency).

variants:
  qpong      : Q-PONG predictor (commitment-conditioned, physics + data loss)
  data_only  : ablation "Data-only" (same data, data loss only)
  pinn_opf   : PINN-OPF baseline (all units committed, physics + data loss)

    python scripts/train_pinn.py --system ieee57 --variant qpong data_only --seeds 0 1 2 --n-jobs 8
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from _common import base_parser, config_for, smoke_defaults

from qpong.data import day_splits
from qpong.experiments import model_path
from qpong.grid import GridModel
from qpong.stage3_pinn.dataset import generate
from qpong.stage3_pinn.train import save_model, train_pinn
from qpong.utils import get_logger, save_json

log = get_logger("train_pinn")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--variant", nargs="+", default=["qpong", "data_only", "pinn_opf"])
    args = smoke_defaults(ap.parse_args())
    for system in args.system:
        cfg = config_for(system, args)
        grid = GridModel.from_config(cfg)
        sp = day_splits(stride=cfg["data"]["test_day_stride"], n_val=cfg["data"]["n_val_days"])
        n = cfg["stage3"]["n_train_samples"]
        cache = {}
        for variant in args.variant:
            all_on = variant == "pinn_opf"
            if all_on not in cache:
                dpath = Path(cfg["_repo_root"]) / cfg["experiment"]["results_dir"] / "pinn_data" / \
                    f"{system}_{'allon' if all_on else 'uc'}.npz"
                if dpath.exists():
                    z = np.load(dpath)
                    cache[all_on] = (z["X"], z["Y"], z["Xv"], z["Yv"])
                else:
                    X, Y, f1 = generate(cfg, sp["train"], n, seed=1, n_jobs=args.n_jobs, all_on=all_on)
                    Xv, Yv, f2 = generate(cfg, sp["val"], max(n // 10, 50), seed=2, n_jobs=args.n_jobs,
                                          all_on=all_on)
                    log.info("%s: %d train / %d val samples (%d / %d OPF failures skipped)", system,
                             len(X), len(Xv), f1, f2)
                    dpath.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dpath, X=X, Y=Y, Xv=Xv, Yv=Yv)
                    cache[all_on] = (X, Y, Xv, Yv)
            X, Y, Xv, Yv = cache[all_on]
            for seed in (args.seeds or cfg["experiment"]["seeds"]):
                model, info = train_pinn(grid, cfg, X, Y, Xv, Yv, physics=(variant != "data_only"), seed=seed)
                p = model_path(cfg, variant, seed)
                save_model(model, info, p)
                save_json(info, p.with_suffix(".json"))
                log.info("saved %s (%d epochs, val %.3e)", p, info["epochs"], info["best_val"])


if __name__ == "__main__":
    main()
