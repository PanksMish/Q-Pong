"""Common 15-min time index for the study year (naive local clock, 35,040 steps)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def year_index(year: int = 2018, minutes: int = 15) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:45", freq=f"{minutes}min")


def to_year_grid(s: pd.Series, year: int = 2018, minutes: int = 15) -> pd.Series:
    """Reindex a series onto the year grid; gaps (e.g. DST spring-forward) are
    filled by time interpolation and the edges by nearest value."""
    s = s[~s.index.duplicated(keep="first")].sort_index()
    idx = year_index(year, minutes)
    out = s.reindex(s.index.union(idx)).interpolate("time").reindex(idx)
    return out.bfill().ffill()


def energy_preserving_upsample(values: np.ndarray, factor: int) -> np.ndarray:
    """Upsample interval-average values by an integer factor while keeping the
    mean of every original interval (energy) unchanged.

    A linear interpolation between interval mid-points gives the shape; each
    block of ``factor`` sub-values is then shifted so its mean equals the
    original interval average, and clipped at zero with re-normalisation.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    x_old = np.arange(n) + 0.5
    x_new = (np.arange(n * factor) + 0.5) / factor
    shape = np.interp(x_new, x_old, v)
    blocks = shape.reshape(n, factor)
    blocks = blocks - blocks.mean(axis=1, keepdims=True) + v[:, None]
    blocks = np.clip(blocks, 0.0, None)
    m = blocks.mean(axis=1, keepdims=True)
    scale = np.divide(v[:, None], m, out=np.zeros_like(m), where=m > 0)
    return (blocks * scale).ravel()
