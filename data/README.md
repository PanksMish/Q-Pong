# Data

`raw/` is not committed (see `.gitignore`). Expected files:

```
raw/pecanstreet/15minute_data_austin.csv   Pecan Street Dataport export (licensed, not redistributable)
raw/nsrdb/nsrdb_austin_2018.csv            NREL NSRDB PSM3, Austin TX, 2018, 30-min, GHI/DNI/DHI/Temperature
raw/ercot/DAMLZHBSPP_2018.xlsx             ERCOT NP4-180-ER, DAM hub prices 2018
raw/ercot/RTMLZHBSPP_2018.xlsx             ERCOT NP6-785-ER, RTM hub prices 2018
raw/egrid/egrid2023_data.xlsx              EPA eGRID2023 (sheet SRL23, column SRCO2RTA, subregion ERCT)
```

## Pecan Street

Export the Austin 15-minute table for 1 Jan – 31 Dec 2018 with at least the columns
`dataid, local_15min, grid, solar, solar2`. Household consumption is computed as
`grid + solar + solar2`; the 25 households with the most complete records are aggregated
(the selected `dataid`s are written to `processed/profiles_2018.meta.json`).

## Processing

`python scripts/prepare_data.py --rebuild` aligns all sources on a 15-minute 2018 grid
(energy-preserving interpolation for the 30-min NSRDB data, hourly DAM prices held within the hour,
DST gaps interpolated) and writes `processed/profiles_2018.csv.gz` plus a metadata file.
The derived normalized profiles may be shared; the Dataport export may not.

Day split (`qpong/data/splits.py`): test days = day-of-year 12, 24, …, 360 (30 days);
validation = 30 of the remaining days; training = the other 305 days.
