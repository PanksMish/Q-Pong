"""SYNTHETIC profiles for smoke tests and CI ONLY.

These are NOT the data of the paper. They let the pipeline, the tests and the
GitHub Actions workflow run without the licensed Pecan Street export or the
NREL API key. Every result file produced from them carries
``data_source = synthetic``; never report such numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .timeindex import year_index


def synthetic_profiles(year: int = 2018, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = year_index(year)
    doy = idx.dayofyear.to_numpy()
    h = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    season = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 200) / 365)          # summer peak
    daily = 0.6 + 0.25 * np.exp(-((h - 19) / 3) ** 2) + 0.15 * np.exp(-((h - 8) / 2) ** 2)
    load = 25 * (1.0 + 0.5 * season) * daily * (1 + 0.05 * rng.standard_normal(len(idx)))
    day_len = 12 + 2 * np.cos(2 * np.pi * (doy - 172) / 365)
    sun = np.clip(np.cos(np.pi * (h - 13) / day_len), 0, None) ** 1.3
    cloud = np.repeat(np.clip(rng.beta(4, 1.5, size=365), 0.1, 1), 96)[: len(idx)]
    pv = sun * cloud * (1 + 0.03 * rng.standard_normal(len(idx)))
    pv = np.clip(pv, 0, None)
    base_price = 25 + 15 * season + 10 * np.exp(-((h - 17) / 2.5) ** 2)
    price_da = base_price * (1 + 0.05 * rng.standard_normal(len(idx)))
    price_rt = price_da * (1 + 0.15 * rng.standard_normal(len(idx)))
    df = pd.DataFrame({
        "load_kw": np.clip(load, 1, None),
        "pv_rooftop_kw": 20 * pv,
        "pv_avail": pv / pv.max(),
        "ghi": 1000 * pv, "dni": 800 * pv,
        "price_da": price_da, "price_rt": np.clip(price_rt, -10, None),
    }, index=idx)
    df.attrs["data_source"] = "synthetic"
    return df
