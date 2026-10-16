"""Pecan Street Dataport loader (Austin, 15-min, residential load and rooftop PV).

Pecan Street data are licensed and cannot be redistributed. Export the Austin
15-minute table from Dataport (https://www.pecanstreet.org/dataport/) for the
year 2018 and place it at ``data.pecanstreet_file``. Expected columns:

    dataid, local_15min, grid, solar, solar2   (kW, interval average)

Household consumption = grid + solar + solar2 (``grid`` is net import).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .timeindex import to_year_grid


def _parse_local(ts: pd.Series) -> pd.DatetimeIndex:
    # Dataport stamps look like "2018-01-01 00:00:00-06"; keep the local wall clock.
    t = pd.to_datetime(ts.astype(str).str.slice(0, 19), errors="coerce")
    return pd.DatetimeIndex(t)


def load_pecanstreet(path, n_homes: int = 25, year: int = 2018) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "local_15min" not in df.columns:
        raise ValueError(f"{path}: expected a 'local_15min' column (Dataport 15-min export)")
    for c in ("grid", "solar", "solar2"):
        if c not in df.columns:
            df[c] = 0.0
    df[["grid", "solar", "solar2"]] = df[["grid", "solar", "solar2"]].fillna(0.0)
    df["t"] = _parse_local(df["local_15min"])
    df = df[df["t"].dt.year == year]

    # choose the n_homes households with the most complete records (deterministic)
    counts = df.groupby("dataid")["t"].count().sort_values(ascending=False, kind="stable")
    homes = sorted(counts.index[:n_homes].tolist())
    df = df[df["dataid"].isin(homes)]

    df["pv"] = (df["solar"] + df["solar2"]).clip(lower=0.0)
    df["load"] = (df["grid"] + df["pv"]).clip(lower=0.0)
    agg = df.groupby("t")[["load", "pv"]].sum()
    out = pd.DataFrame({
        "load_kw": to_year_grid(agg["load"], year),
        "pv_rooftop_kw": to_year_grid(agg["pv"], year),
    })
    out.attrs["homes"] = homes
    return out.astype(np.float64)
