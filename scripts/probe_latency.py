"""Probe LM Studio latency and reasoning-control options.

    python scripts\probe_latency.py

Measures a plain chat call against several variants that try to suppress the
model's <think> block, since a truncated think block produces an empty answer
and a 60% parse rate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

import _bootstrap  # noqa: F401

from qlab_rl.config import load_policy
from qlab_rl.llm.client import LLMClient
from qlab_rl.policies.llm import parse_action

SYSTEM = "You are a trading risk manager. Answer in the exact format asked."

USER = """Date: 2024-03-05   Close: 512.30

Market state:
  return_1d            +0.42%
  return_5d            -1.10%
  return_20d           +2.85%
  vol_20d              +14.20%
  drawdown_20d         -0.90%
  exposure             +0.00

Choose exactly one position exposure:
  SHORT_ALL = -1.00, SHORT_HALF = -0.50, FLAT = 0.00, LONG_HALF = +0.50, LONG_ALL = +1.00

Reply in exactly this format and nothing else:
ACTION: <one of the option names>
REASON: <one short sentence>
"""

VARIANTS = [
    ("baseline-max160", {}),
    ("max160-think-off", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("max160-reasoning-none", {"reasoning_effort": "none"}),
    ("max160-extra-no-think-tag", {"_extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}),
    ("max160-reasoning-low", {"reasoning_effort": "low"}),
]


def raw_call(base_url: str, model: str, extra: dict, max_tokens: int, temperature: float) -> tuple[str, float, str]:
    """POST straight to the endpoint so unknown params are passed through."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        **extra,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
    except Exception as exc:
        return f"<ERROR> {exc}", time.time() - started, ""
    elapsed = time.time() - started
    text = (body["choices"][0].get("message") or {}).get("content") or ""
    usage = body.get("usage") or {}
    return text, elapsed, str(usage)


def cache_experiment(base_url: str, model: str, n: int = 6) -> None:
    """Is the 7s/call from prefill? Repeat one prompt vs vary it."""
    print("\nprefix-cache check (same prompt vs varying prompt)")
    print("-" * 72)

    def one(user_text: str) -> float:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.0,
            "max_tokens": 160,
            "reasoning_effort": "none",
        }
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            json.loads(resp.read().decode())
        return time.time() - t0

    first = one(USER)
    same = [one(USER) for _ in range(n)]
    varied = [one(USER.replace("2024-03-05", f"2024-03-{5 + i:02d}")
                     .replace("+512.30", f"+{512.30 + i:.2f}")
                     .replace("+0.42%", f"+{0.42 + i * 0.1:.2f}%")) for i in range(n)]

    print(f"  first call (cold)      {first:6.2f}s")
    print(f"  identical prompt  mean {statistics.mean(same):6.2f}s   <- prefix cached")
    print(f"  varying prompt   mean {statistics.mean(varied):6.2f}s   <- full prefill")
    print(f"  penalty for varying      {statistics.mean(varied) - statistics.mean(same):6.2f}s/call")


def main() -> int:
    cfg = load_policy()["llm"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=cfg["base_url"])
    ap.add_argument("--model", default=cfg["model"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    client = LLMClient(base_url=args.base_url, model=args.model)
    model = client.resolve_model()
    print(f"model: {model}\n")

    print(f"{'variant':<30} {'mean_s':>7} {'min_s':>7} {'max_s':>7} "
          f"{'ok':>4} {'think':>6} {'out_tok':>8}  action")
    print("-" * 100)

    best = None
    for name, extra in VARIANTS:
        times, oks, thinks, outs, action = [], 0, 0, [], "n/a"
        for _ in range(args.repeats):
            text, elapsed, usage = raw_call(
                args.base_url, model, extra, cfg["max_tokens"], cfg["temperature"]
            )
            times.append(elapsed)
            exposure, _ = parse_action(text, 5)
            if exposure is not None:
                oks += 1
                action = f"{exposure:+.2f}"
            if "<think>" in text or "</think>" in text:
                thinks += 1
            try:
                outs.append(json.loads(usage).get("completion_tokens", 0))
            except Exception:
                pass

        mean = statistics.mean(times)
        if oks == args.repeats and (best is None or mean < best[1]):
            best = (name, mean)
        print(f"{name:<30} {mean:>7.2f} {min(times):>7.2f} {max(times):>7.2f} "
              f"{oks:>2}/{args.repeats} {thinks:>4}/{args.repeats} "
              f"{(statistics.mean(outs) if outs else 0):>8.0f}  {action}")

    if best is None:
        print("\nno variant parsed cleanly - raise max_tokens or change the model")
    else:
        name, mean = best
        print(f"\nbest working variant: {name} at {mean:.2f}s/call")
        print(f"  570 decisions  ~ {570 * mean / 60:.0f} min per episode")

    cache_experiment(args.base_url, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())