"""Ask the hosted model for a single trading decision, print it, done.

    python scripts\ask_model.py
    python scripts\ask_model.py --repeat 3
    python scripts\ask_model.py --json outputs\decision.json
    python scripts\ask_model.py --free   (no options, just "what do you do?")

This is the smoke test. If it prints a sensible decision, the whole stack works.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qlab_rl.config import load_policy, load_providers, resolve_provider
from qlab_rl.llm.client import LLMClient, LLMError
from qlab_rl.policies.actions import label_for, labels
from qlab_rl.policies.llm import build_user_message, parse_action

DEMO_OBS = [0.0125, 0.0340, 0.0712, 0.1520, -0.0180, 0.0]
DEMO_DATE = "2021-06-15"
DEMO_PRICE = 425.18


def main() -> int:
    cfg = load_policy()
    llm_cfg, pol_cfg = cfg["llm"], cfg["policy"]
    n_actions = pol_cfg["n_actions"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=llm_cfg.get("provider", "lmstudio"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--free", action="store_true", help="skip the option list, open-ended question")
    ap.add_argument("--prompt", default=None, help="override the whole user message")
    ap.add_argument("--json", default=None, help="write results to this path")
    args = ap.parse_args()
    try:
        prov_cfg = resolve_provider(
            args.provider,
            llm_cfg,
            load_providers(),
            {
                "base_url": args.base_url,
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "timeout": args.timeout,
            },
        )
    except LLMError as exc:
        print(f"[FAIL] {exc}")
        return 1
    client = LLMClient.from_provider(prov_cfg)

    try:
        model = client.resolve_model()
    except LLMError as exc:
        print(f"[FAIL] {exc}")
        print("Run: python scripts\\check_provider.py --provider " + args.provider)
        return 1

    print(f"provider: {args.provider}")
    print(f"model:    {model}")
    print(f"endpoint: {prov_cfg['base_url']}")
    print(f"actions:  {', '.join(labels(n_actions))}\n")

    if args.free:
        user = args.prompt or (
            "Close is 425.18. 5-day return is +3.40%, 20-day return is +7.12%, "
            "annualised volatility is 15.2%. What do you do next and why?"
        )
        system = pol_cfg["system_prompt"]
    else:
        user = args.prompt or build_user_message(DEMO_OBS, DEMO_DATE, DEMO_PRICE, n_actions)
        system = pol_cfg["system_prompt"]

    print("=" * 70)
    print("PROMPT")
    print("=" * 70)
    print(user)
    print()

    results = []
    parsed_count = 0
    for k in range(args.repeat):
        try:
            text = client.text_of(client.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]))
        except LLMError as exc:
            print(f"[FAIL] call {k + 1}/{args.repeat}: {exc}")
            return 1

        exposure, label = parse_action(text, n_actions)
        ok = exposure is not None
        parsed_count += int(ok)
        results.append({"raw": text, "label": label, "exposure": exposure})

        print("=" * 70)
        print(f"RESPONSE {k + 1}/{args.repeat}")
        print("=" * 70)
        print(text.strip() or "(empty)")
        print(f"\n-> parsed: {label_for(n_actions, exposure) if ok else 'UNPARSED (would fall back to FLAT)'}")
        print()

    print("=" * 70)
    print(f"parse rate: {parsed_count}/{args.repeat}")
    if args.repeat > 1:
        choices = {r["label"] for r in results}
        print(f"stability : {'stable' if len(choices) == 1 else 'VARIES -> ' + str(sorted(choices))}")
    print("=" * 70)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"model": model, "endpoint": args.base_url, "prompt": user, "results": results},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"written: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())