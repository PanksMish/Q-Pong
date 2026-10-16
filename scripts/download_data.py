#!/usr/bin/env python
"""Fetch the public data sources (Section V-B) into data/raw/.

    python scripts/download_data.py nsrdb --api-key $NREL_API_KEY --email you@uni.edu
    python scripts/download_data.py egrid      # prints where to download the workbook
    python scripts/download_data.py ercot      # prints the two report pages to download from
    python scripts/download_data.py pecanstreet  # licensed: prints export instructions

Pecan Street Dataport data are licensed and cannot be redistributed; export
the Austin 15-minute table for 2018 (columns dataid, local_15min, grid, solar,
solar2) and save it as data/raw/pecanstreet/15minute_data_austin.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

EGRID_PAGE = "https://www.epa.gov/egrid/download-data"
ERCOT_PAGES = {
    "DAM (NP4-180-ER, Historical DAM Load Zone and Hub Prices)":
        "https://www.ercot.com/mp/data-products/data-product-details?id=NP4-180-ER",
    "RTM (NP6-785-ER, Historical RTM Load Zone and Hub Prices)":
        "https://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-ER",
}


def nsrdb(api_key, email, lat=30.2672, lon=-97.7431, year=2018):
    """Uses pvlib's NSRDB client (pip install pvlib), which tracks the NREL API endpoint."""
    try:
        from pvlib.iotools import get_psm3
    except ImportError:
        sys.exit("pip install pvlib  (or download the PSM3 CSV for Austin, 2018, 30-min from "
                 "https://nsrdb.nrel.gov/data-viewer)")
    data, meta = get_psm3(lat, lon, api_key, email, names=str(year), interval=30,
                          attributes=("ghi", "dni", "dhi", "air_temperature"), leap_day=False,
                          map_variables=False)
    out = RAW / "nsrdb" / "nsrdb_austin_2018.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = data.copy()
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    for c, k in (("Year", "year"), ("Month", "month"), ("Day", "day"), ("Hour", "hour"), ("Minute", "minute")):
        if c not in df:
            df[c] = getattr(df.index, k)
    with open(out, "w") as f:
        f.write(",".join(map(str, meta.keys())) + "\n" + ",".join(map(str, meta.values())) + "\n")
        df.to_csv(f, index=False)
    print("wrote", out)


def egrid():
    print(f"Download the eGRID2023 data workbook (xlsx) from {EGRID_PAGE}")
    print(f"and save it as {RAW / 'egrid' / 'egrid2023_data.xlsx'}")


def ercot():
    print("Download the 2018 workbooks from:")
    for k, v in ERCOT_PAGES.items():
        print(f"  {k}\n    {v}")
    print(f"and save them as {RAW / 'ercot' / 'DAMLZHBSPP_2018.xlsx'} and {RAW / 'ercot' / 'RTMLZHBSPP_2018.xlsx'}")


def pecanstreet():
    print(__doc__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["nsrdb", "egrid", "ercot", "pecanstreet"])
    ap.add_argument("--api-key")
    ap.add_argument("--email")
    a = ap.parse_args()
    if a.source == "nsrdb":
        if not (a.api_key and a.email):
            sys.exit("--api-key and --email are required (free key: https://developer.nrel.gov/signup/)")
        nsrdb(a.api_key, a.email)
    else:
        globals()[a.source]()
