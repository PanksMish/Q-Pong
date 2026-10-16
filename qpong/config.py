"""Configuration loading: default.yaml deep-merged with a system file and CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(system: str = "ieee30", extra: list[str] | None = None,
                overrides: list[str] | None = None) -> dict:
    """Load configuration.

    Parameters
    ----------
    system : name of a system file in ``configs/`` (``ieee30``, ``ieee57``, ``ieee118``).
    extra : additional YAML files merged in order (e.g. ``["noise"]``).
    overrides : ``key.sub=value`` strings parsed with YAML (e.g. ``stage2.maxiter=50``).
    """
    with open(CONFIG_DIR / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    for name in [system] + list(extra or []):
        path = Path(name)
        if not path.suffix:
            path = CONFIG_DIR / f"{name}.yaml"
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(val))
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve_path(cfg: dict, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else Path(cfg["_repo_root"]) / path
