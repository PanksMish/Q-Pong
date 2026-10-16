"""EPA eGRID 2023 subregion CO2 output emission rate (ERCT).

Download eGRID2023 data (xlsx) from https://www.epa.gov/egrid/download-data and
place it at ``data.egrid_file``. The subregion sheet (``SRL23``) carries the
annual CO2 total output emission rate in column ``SRCO2RTA`` (lb/MWh).
The carbon cost is C_carbon,t = gamma_ERCT * sum_i P_i,t * dt.
"""
from __future__ import annotations

import pandas as pd

LB_TO_TONNE = 0.000453592


def load_egrid_rate(path, subregion: str = "ERCT", sheet: str | None = None,
                    column: str = "SRCO2RTA") -> float:
    """Return the subregion CO2 rate in tonne/MWh."""
    xls = pd.ExcelFile(path)
    sheet = sheet or next(s for s in xls.sheet_names if s.upper().startswith("SRL"))
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    # the header row is the first row that contains the column code
    hdr = next(i for i in range(min(10, len(raw))) if column in raw.iloc[i].astype(str).tolist())
    df = pd.read_excel(path, sheet_name=sheet, header=hdr)
    row = df[df["SUBRGN"].astype(str).str.upper() == subregion.upper()]
    if row.empty:
        raise ValueError(f"subregion {subregion} not found in {path}:{sheet}")
    return float(row[column].iloc[0]) * LB_TO_TONNE
