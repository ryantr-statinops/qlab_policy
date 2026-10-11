"""Doctor: is LM Studio up, and is a chat model loaded?

    python scripts\check_lmstudio.py

Exit code 0 = ready, 1 = not ready.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from qlab_rl.config import load_policy
from qlab_rl.llm.client import LLMClient, LLMError

EMBED_HINTS = ("embed", "nomic", "bge", "gte", "minilm", "e5")


def main() -> int:
    ap = argparse.ArgumentParser()
    cfg = load_policy()["llm"]
    ap.add_argument("--base-url", default=cfg["base_url"])
    ap.add_argument("--timeout", type=int, default=5)
    args = ap.parse_args()

    print(f"base_url: {args.base_url}")

    client = LLMClient(base_url=args.base_url, timeout=args.timeout, retries=0)
    try:
        models = client.list_models()
    except LLMError as exc:
        print("\n[FAIL] server not reachable.")
        print(f"  {exc}")
        print("\nHow to fix:")
        print("  1. Open LM Studio (C:\\Users\\Admin\\AppData\\Local\\Programs\\LM Studio\\LM Studio.exe)")
        print("  2. My Models -> download a chat model ~4B (Qwen3-4B-Instruct Q4_K_M fits 4GB VRAM)")
        print("  3. Load the model")
        print("  4. Developer tab -> Start Server")
        print("\nCLI alternative:")
        print("  lms pull qwen3-4b-instruct-q4_k_m")
        print("  lms load qwen3-4b-instruct --context-length 8192")
        print("  lms server start")
        return 1

    print(f"[OK] server responded, {len(models)} model(s) reported.\n")
    chat_models = []
    for m in models:
        mid = m.get("id", "?")
        is_embed = any(h in mid.lower() for h in EMBED_HINTS)
        kind = "EMBEDDING" if is_embed else "chat"
        ctx = m.get("max_context_length") or m.get("context_length") or "?"
        print(f"  - {mid}  [{kind}, ctx={ctx}]")
        if not is_embed:
            chat_models.append(mid)

    if not chat_models:
        print("\n[FAIL] only embedding models are loaded. An embedding model cannot")
        print("       answer chat prompts - download and load a chat model instead.")
        return 1

    print(f"\n[OK] ready to use: {chat_models[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())