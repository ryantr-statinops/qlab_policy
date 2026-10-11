"""Config loading via stdlib tomllib. No third-party dependencies."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def load(name: str) -> dict[str, Any]:
    """Load configs/<name>.toml and return the parsed dict."""
    path = CONFIG_DIR / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_data() -> dict[str, Any]:
    return load("data")["data"]


def load_env() -> dict[str, Any]:
    cfg = load("env")
    return {"env": cfg["env"], "split": cfg["split"], "state": cfg["state"]}


def load_policy() -> dict[str, Any]:
    cfg = load("policy")
    return {"llm": cfg["llm"], "policy": cfg["policy"], "prompt": cfg["prompt"]}


def resolve(path_like: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p