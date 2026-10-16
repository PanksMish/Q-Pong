#!/usr/bin/env python
"""Build data/processed/profiles_2018.csv.gz and check the scenarios against each case.

The check reports, per system, penetration and stress class, the share of test
intervals whose net load exceeds the installed capacity (with reserve) or
falls below the minimum stable generation of the encoded units - intervals
that no method can serve feasibly. Run it before the experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import base_parser, config_for

from qpong.data import build_profiles, day_splits, load_profiles, make_scenario
from qpong.grid import GridModel


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    rows = []
    for system in args.system:
        cfg = config_for(system, args)
        if args.rebuild:
            build_profiles(cfg)
        prof, meta = load_profiles(cfg)
        grid = GridModel.from_config(cfg)
        cap = grid.units.pmax.sum()
        res = cfg["grid"]["reserve_fraction"]
        for pen in cfg["data"]["penetrations"]:
            for stress in ("normal", "high_demand", "low_renewables", "combined"):
                nl = []
                for d in day_splits()["test"]:
                    sc = make_scenario(grid, prof, meta, cfg, d, pen, stress)
                    nl.append(sc.pd.sum(1) - sc.ppv.sum(1))
                nl = np.concatenate(nl)
                rows.append({"system": system, "penetration": pen, "stress": stress,
                             "netload_min_MW": nl.min(), "netload_max_MW": nl.max(), "capacity_MW": cap,
                             "pct_over_capacity_with_reserve": 100 * np.mean(nl * (1 + res) > cap),
                             "pct_over_capacity": 100 * np.mean(nl > cap),
                             "data_source": meta["data_source"]})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    out = config_for(args.system[0], args)["experiment"]["results_dir"]
    df.to_csv(f"{out}/scenario_check.csv", index=False)


if __name__ == "__main__":
    main()
