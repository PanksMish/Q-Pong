"""Assemble the processed 2018 profile table from the four public data sources."""
from __future__ import annotations

import json

import pandas as pd

from ..config import resolve_path
from ..utils import get_logger
from .egrid import load_egrid_rate
from .ercot import load_ercot
from .nsrdb import load_nsrdb
from .pecanstreet import load_pecanstreet
from .synthetic import synthetic_profiles

log = get_logger("qpong.data")


def _paths(cfg):
    d = cfg["data"]
    proc = resolve_path(cfg, d["processed_file"]).with_suffix(".csv.gz")
    meta = proc.with_name(proc.name.replace(".csv.gz", ".meta.json"))
    return proc, meta


def build_profiles(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    year = d["year"]
    proc, meta_path = _paths(cfg)
    proc.parent.mkdir(parents=True, exist_ok=True)

    if d["source"] == "synthetic":
        df = synthetic_profiles(year)
        meta = {"data_source": "synthetic", "co2_t_per_mwh": 0.40}
        log.warning("Using SYNTHETIC profiles (smoke test only, not the paper data).")
    else:
        ps = load_pecanstreet(resolve_path(cfg, d["pecanstreet_file"]), d["pecanstreet_n_homes"], year)
        ns = load_nsrdb(resolve_path(cfg, d["nsrdb_file"]), year)
        er = load_ercot(resolve_path(cfg, d["ercot_dam_file"]), resolve_path(cfg, d["ercot_rtm_file"]),
                        d["ercot_hubs"], year)
        df = pd.concat([ps, ns, er], axis=1)
        co2 = d.get("egrid_co2_lb_per_mwh")
        rate = (co2 * 0.000453592) if co2 else load_egrid_rate(resolve_path(cfg, d["egrid_file"]),
                                                                d["egrid_subregion"])
        meta = {"data_source": "real", "co2_t_per_mwh": rate,
                "pecanstreet_homes": ps.attrs.get("homes", [])}
    if df.isna().any().any():
        log.warning("NaNs after alignment; forward/back filling %d values", int(df.isna().sum().sum()))
        df = df.ffill().bfill()
    df.to_csv(proc, compression="gzip", index_label="t")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("wrote %s (%d rows)", proc, len(df))
    return df


def load_profiles(cfg: dict) -> tuple[pd.DataFrame, dict]:
    proc, meta_path = _paths(cfg)
    if not proc.exists():
        build_profiles(cfg)
    df = pd.read_csv(proc, index_col="t", parse_dates=["t"])
    with open(meta_path) as f:
        meta = json.load(f)
    return df, meta
