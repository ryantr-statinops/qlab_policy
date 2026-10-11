"""Evaluate on the test split: LLM policy vs every reference policy.

    python scripts\eval.py
    python scripts\eval.py --split test --max-steps 60
"""

from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401

from qlab_rl.config import load_data, load_env, load_policy, resolve
from qlab_rl.data.provider import YahooCsvProvider
from qlab_rl.llm.client import LLMClient, LLMError
from qlab_rl.policies import rule
from qlab_rl.policies.llm import LLMPolicy
from qlab_rl.runner import build_env, load_segment, run_policy


def fmt(m: dict) -> str:
    def pct(x):
        return "   n/a" if x != x else f"{x * 100:7.2f}%"

    def num(x):
        return "   n/a" if x != x else f"{x:6.2f}"

    return (
        f"ret={pct(m['total_return'])} sharpe={num(m['sharpe'])} "
        f"cagr={pct(m['cagr'])} maxDD={pct(m['max_drawdown'])} "
        f"PF={num(m['profit_factor'])} calmar={num(m['calmar'])} "
        f"flat={pct(m['flat_ratio'])} expo={num(m['mean_exposure'])}"
    )


def main() -> int:
    data_cfg = load_data()
    env_cfg = load_env()
    pol_cfg = load_policy()
    frac = env_cfg["split"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=data_cfg["ticker"])
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--actions", type=int, default=pol_cfg["policy"]["n_actions"])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--base-url", default=pol_cfg["llm"]["base_url"])
    ap.add_argument("--model", default=pol_cfg["llm"]["model"])
    args = ap.parse_args()

    cache = resolve(data_cfg["cache_dir"])
    if not (cache / f"{args.ticker}.csv").exists():
        print(f"[FAIL] no data. Run: python scripts\\fetch_data.py")
        return 1

    bars = YahooCsvProvider(args.ticker, "", None, cache).load()
    segment = load_segment(bars, args.split, frac["train"], frac["val"], frac["test"])

    print(f"ticker {args.ticker}  split={args.split}  "
          f"{segment[0].date} -> {segment[-1].date}  n={len(segment)}\n")

    results = []
    for name in rule.ALL_NAMES:
        env = build_env(segment, name)
        m = run_policy(env, rule.make(name), max_steps=args.max_steps)
        results.append((name, m))
        print(f"  {name:<18} {fmt(m)}")

    if not args.skip_llm:
        client = LLMClient.from_config(
            pol_cfg["llm"],
            base_url=args.base_url,
            model=args.model,
        )
        try:
            model = client.resolve_model()
            model_short = model.split("/")[-1]
            env = build_env(segment, f"llm/{model_short}")
            policy = LLMPolicy(
                client, n_actions=args.actions,
                system_prompt=pol_cfg["policy"]["system_prompt"],
            )
            started = time.time()
            m = run_policy(env, policy, max_steps=args.max_steps)
            elapsed = time.time() - started
            name = f"llm:{model_short}"
            results.append((name, m))
            print(f"\n  {name:<18} {fmt(m)}")
            print(f"  {'':<18} parse_rate={m['parse_rate'] * 100:.1f}%  "
                  f"{m['steps']} calls in {elapsed:.1f}s "
                  f"({m['steps'] / elapsed if elapsed else 0:.2f} steps/s)")
            hist = m["action_hist"]
            if hist:
                print(f"  {'':<18} actions: "
                      + "  ".join(f"{k:+.1f}:{v}" for k, v in sorted(hist.items())))
        except LLMError as exc:
            print(f"\n  [LLM skipped] {exc}")

    ranked = sorted(
        [(n, m) for n, m in results if m["sharpe"] == m["sharpe"]],
        key=lambda kv: kv[1]["sharpe"],
        reverse=True,
    )
    print(f"\nranked by sharpe on {args.split}:")
    for name, m in ranked:
        print(f"  {m['sharpe']:6.2f}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())