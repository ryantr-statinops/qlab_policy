"""Config loading via stdlib tomllib. No third-party dependencies."""

from __future__ import annotations

import os
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


def load_providers() -> dict[str, dict[str, object]]:
    """Load configs/providers.toml; each top-level table is one provider preset."""
    return load("providers")


def resolve_provider(
    name: str,
    base_cfg: dict[str, Any],
    presets: dict[str, dict[str, Any]],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge one provider preset. Precedence: overrides > preset > base_cfg.

    Cloud presets declare `api_key_env`; the key is read from that environment
    variable only, never from a file. A missing key raises LLMError naming the
    variable (the key value is never included in the message).
    """
    from .llm.client import LLMError

    if name not in presets:
        raise LLMError(f"unknown provider {name!r}; choose from {sorted(presets)}")
    preset = presets[name]
    ov = overrides or {}

    def pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in ov and ov[key] is not None:
                return ov[key]
        return default

    api_key_env = str(preset.get("api_key_env", "") or "")
    api_key = ""
    if api_key_env:
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise LLMError(
                f"provider {name!r} needs an API key: set ${api_key_env} in the environment."
            )

    timeout = pick("timeout", "timeout_sec", default=None)
    if timeout is None:
        timeout = preset.get("timeout_sec", base_cfg.get("timeout_sec", 60))

    model = pick("model", default=None)
    if model is None:
        model = preset.get("model") or base_cfg.get("model", "")

    return {
        "provider": name,
        "base_url": pick("base_url", default=None)
        or preset.get("base_url")
        or base_cfg.get("base_url"),
        "model": model,
        "api_key": api_key,
        "api_key_env": api_key_env,
        "auto_detect": bool(preset.get("auto_detect", True)),
        "extra_headers": dict(preset.get("extra_headers", {}) or {}),
        "temperature": pick("temperature", default=base_cfg.get("temperature", 0.0)),
        "max_tokens": pick("max_tokens", default=base_cfg.get("max_tokens", 160)),
        "timeout_sec": timeout,
        "retries": pick("retries", default=base_cfg.get("retries", 2)),
        "reasoning_effort": pick("reasoning_effort", default=base_cfg.get("reasoning_effort")),
    }


def resolve(path_like: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p