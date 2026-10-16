"""NREL NSRDB loader and irradiance-to-PV conversion (Austin, TX, 2018).

Download (requires a free NREL API key)::

    python scripts/download_data.py nsrdb --api-key $NREL_API_KEY --email you@example.org

The PSM3 CSV has two metadata lines followed by a header with
Year, Month, Day, Hour, Minute, GHI, DNI, DHI, Temperature, ...
Time stamps are local standard time (UTC-6 for Austin).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .timeindex import energy_preserving_upsample, year_index

AUSTIN_LAT, AUSTIN_LON = 30.2672, -97.7431


def read_psm3(path) -> tuple[pd.DataFrame, dict]:
    meta = pd.read_csv(path, nrows=1).iloc[0].to_dict()
    df = pd.read_csv(path, skiprows=2)
    df.index = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    return df, meta


def solar_zenith_azimuth(times: pd.DatetimeIndex, lat: float, lon: float, tz_hours: float = -6.0):
    """Solar position (Spencer 1971 / NOAA approximations), radians."""
    doy = times.dayofyear.values
    hour = times.hour.values + times.minute.values / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    tst = hour * 60 + eqt + 4 * lon - 60 * tz_hours
    ha = np.deg2rad(tst / 4.0 - 180.0)
    phi = np.deg2rad(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    zen = np.arccos(np.clip(cosz, -1, 1))
    az = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(phi) - np.tan(decl) * np.cos(phi)) + np.pi
    return zen, az


def pv_availability(df: pd.DataFrame, lat=AUSTIN_LAT, lon=AUSTIN_LON, tilt_deg=None,
                    albedo=0.2, gamma_t=-0.004, noct=45.0) -> np.ndarray:
    """Normalized PV availability in [0, 1] from GHI, DNI (and DHI, temperature).

    Plane of array: south-facing, tilt = latitude, isotropic sky (Liu-Jordan).
    Temperature derate with NOCT cell model. Result normalized to its maximum.
    """
    tilt = np.deg2rad(lat if tilt_deg is None else tilt_deg)
    zen, az = solar_zenith_azimuth(df.index, lat, lon)
    ghi = df["GHI"].to_numpy(float)
    dni = df["DNI"].to_numpy(float)
    dhi = df["DHI"].to_numpy(float) if "DHI" in df else np.clip(ghi - dni * np.cos(zen), 0, None)
    cos_aoi = np.cos(zen) * np.cos(tilt) + np.sin(zen) * np.sin(tilt) * np.cos(az - np.pi)
    poa = (np.clip(dni * cos_aoi, 0, None) + dhi * (1 + np.cos(tilt)) / 2
           + ghi * albedo * (1 - np.cos(tilt)) / 2)
    poa[np.cos(zen) <= 0] = 0.0
    t_air = df["Temperature"].to_numpy(float) if "Temperature" in df else np.full_like(poa, 25.0)
    t_cell = t_air + poa / 800.0 * (noct - 20.0)
    p = poa / 1000.0 * (1 + gamma_t * (t_cell - 25.0))
    p = np.clip(p, 0, None)
    return p / p.max() if p.max() > 0 else p


def load_nsrdb(path, year: int = 2018, minutes: int = 15) -> pd.DataFrame:
    df, _ = read_psm3(path)
    df = df[df.index.year == year]
    step = int(round((df.index[1] - df.index[0]).total_seconds() / 60))
    avail = pv_availability(df)
    cols = {"pv_avail": avail, "ghi": df["GHI"].to_numpy(float), "dni": df["DNI"].to_numpy(float)}
    idx = year_index(year, minutes)
    out = {}
    if step == minutes:
        for k, v in cols.items():
            out[k] = pd.Series(v, index=df.index).reindex(idx).interpolate().bfill().ffill().to_numpy()
    elif step % minutes == 0:
        f = step // minutes
        for k, v in cols.items():
            up = energy_preserving_upsample(v, f)
            out[k] = np.resize(up, len(idx))
    else:
        raise ValueError(f"unsupported NSRDB resolution {step} min")
    res = pd.DataFrame(out, index=idx)
    res["pv_avail"] = res["pv_avail"] / max(res["pv_avail"].max(), 1e-12)
    return res
