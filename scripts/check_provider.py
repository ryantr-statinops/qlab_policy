"""Doctor: is the configured provider reachable, and is auth valid?

    python scripts\\check_provider.py
    python scripts\\check_provider.py --provider openrouter

Exit code 0 = ready, 1 = not ready. API keys are never printed.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from qlab_rl.config import load_policy, load_providers, resolve_provider
from qlab_rl.llm.client import LLMClient, LLMError


def main() -> int:
    base_cfg = load_policy()["llm"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=base_cfg.get("provider", "lmstudio"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    try:
        prov_cfg = resolve_provider(
            args.provider,
            base_cfg,
            load_providers(),
            {"base_url": args.base_url, "model": args.model},
        )
    except LLMError as exc:
        print(f"\n[FAIL] {exc}")
        return 1

    print(f"provider: {args.provider}")
    print(f"base_url: {prov_cfg['base_url']}")
    if prov_cfg["api_key_env"]:
        print(f"api_key : ${prov_cfg['api_key_env']} (set, hidden)")

    client = LLMClient.from_provider(prov_cfg)
    try:
        models = client.list_models()
    except LLMError as exc:
        print(f"\n[FAIL] {exc}")
        return 1

    print(f"[OK] server responded, {len(models)} model(s) reported.\n")
    for m in models[:20]:
        print(f"  - {m.get('id', '?')}")
    if len(models) > 20:
        print(f"  ... and {len(models) - 20} more")

    try:
        model = client.resolve_model()
    except LLMError as exc:
        print(f"\n[FAIL] {exc}")
        return 1
    print(f"\n[OK] ready to use: {model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
