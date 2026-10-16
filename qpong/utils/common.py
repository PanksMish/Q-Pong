"""Seeding, timing and logging helpers."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def get_logger(name: str = "qpong", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


class Timer:
    """Accumulating per-stage wall-clock timer (used for Table IV)."""

    def __init__(self):
        self.t: dict[str, float] = {}

    @contextmanager
    def __call__(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0

    def total(self) -> float:
        return float(sum(self.t.values()))


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)
