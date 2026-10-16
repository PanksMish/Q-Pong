"""ERCOT settlement-point price loader (DAM hourly, RTM 15-min), 2018, four hubs.

Source: ERCOT Market Prices (https://www.ercot.com/mp/data-products):
  * NP4-180-ER  Historical DAM Load Zone and Hub Prices  -> DAMLZHBSPP_2018.xlsx
  * NP6-785-ER  Historical RTM Load Zone and Hub Prices  -> RTMLZHBSPP_2018.xlsx
Each workbook has one sheet per month. Prices enter only Stage I (price
deviation) and Stage IV (settlement), never the network model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .timeindex import to_year_grid


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "settlement_point_name" in df.columns and "settlement_point" not in df.columns:
        df = df.rename(columns={"settlement_point_name": "settlement_point"})
    return df


def _read_all_sheets(path) -> pd.DataFrame:
    if str(path).endswith(".csv"):
        return _norm_cols(pd.read_csv(path))
    sheets = pd.read_excel(path, sheet_name=None)
    return _norm_cols(pd.concat(sheets.values(), ignore_index=True))


def _hour_ending_to_start(date: pd.Series, he: pd.Series) -> pd.Series:
    h = he.astype(str).str.slice(0, 2).astype(int)
    return pd.to_datetime(date) + pd.to_timedelta(h - 1, unit="h")


def load_dam(path, hubs, year: int = 2018) -> pd.DataFrame:
    df = _read_all_sheets(path)
    if "repeated_hour_flag" in df.columns:
        df = df[df["repeated_hour_flag"].astype(str).str.upper() != "Y"]
    df = df[df["settlement_point"].isin(hubs)]
    df["t"] = _hour_ending_to_start(df["delivery_date"], df["hour_ending"])
    out = {}
    for hub in hubs:
        s = df[df["settlement_point"] == hub].set_index("t")["settlement_point_price"].astype(float)
        out[f"price_da_{hub}"] = to_year_grid(s, year)   # hourly -> 15 min (held / interpolated)
    res = pd.DataFrame(out)
    # hold the hourly DAM price constant within the hour (not interpolated)
    hourly = res.resample("1h").first()
    res = hourly.reindex(res.index, method="ffill")
    return res


def load_rtm(path, hubs, year: int = 2018) -> pd.DataFrame:
    df = _read_all_sheets(path)
    if "repeated_hour_flag" in df.columns:
        df = df[df["repeated_hour_flag"].astype(str).str.upper() != "Y"]
    df = df[df["settlement_point"].isin(hubs)]
    start = pd.to_datetime(df["delivery_date"]) + pd.to_timedelta(df["delivery_hour"].astype(int) - 1, unit="h")
    df["t"] = start + pd.to_timedelta((df["delivery_interval"].astype(int) - 1) * 15, unit="min")
    out = {}
    for hub in hubs:
        s = df[df["settlement_point"] == hub].set_index("t")["settlement_point_price"].astype(float)
        out[f"price_rt_{hub}"] = to_year_grid(s, year)
    return pd.DataFrame(out)


def load_ercot(dam_path, rtm_path, hubs, year: int = 2018) -> pd.DataFrame:
    dam = load_dam(dam_path, hubs, year)
    rtm = load_rtm(rtm_path, hubs, year)
    out = pd.concat([dam, rtm], axis=1)
    out["price_da"] = dam.mean(axis=1)
    out["price_rt"] = rtm.mean(axis=1)
    return out.astype(np.float64)
