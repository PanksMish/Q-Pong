"""Train / validation / test day split (Section V-B).

Test days: every 12th day of the year (day-of-year 12, 24, ..., 360 -> 30 days).
The remaining 335 days give 30 validation days (every 11th remaining day,
starting at the 6th) and 305 training days. No test day is seen in training.
"""
from __future__ import annotations


def day_splits(n_days: int = 365, stride: int = 12, n_val: int = 30) -> dict[str, list[int]]:
    test = [d for d in range(stride - 1, n_days, stride)][:30]          # 0-based day indices
    rest = [d for d in range(n_days) if d not in set(test)]
    val = rest[5::11][:n_val]
    train = [d for d in rest if d not in set(val)]
    assert not set(test) & set(train) and not set(test) & set(val)
    return {"train": train, "val": val, "test": test}
