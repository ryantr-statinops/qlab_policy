"""Run the LLM as the policy through real bars. One LLM call per bar.

    python scripts\run_llm.py --episodes 1
    python scripts\run_llm.py --episodes 5 --split test
    python scripts\run_llm.py --max-steps 40          # quick pacing check
    python scripts\run_llm.py --show-prompt           # print a prompt and exit
"""

from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401

from qlab_rl.config import load_data, load_env, load_policy, load_providers, resolve, resolve_provider
from qlab_rl.data.provider import YahooCsvProvider
from qlab_rl.llm.client import LLMClient, LLMError
from qlab_rl.policies.actions import label_for
from qlab_rl.policies.base import PolicyContext
from qlab_rl.policies.llm import LLMPolicy, build_user_message
from qlab_rl.runner import build_env, load_segment, run_policy, write_records


def main() -> int:
    data_cfg = load_data()
    env_cfg = load_env()
    pol_cfg = load_policy()
    frac = env_cfg["split"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=data_cfg["ticker"])
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (pacing check)")
    ap.add_argument("--actions", type=int, default=pol_cfg["policy"]["n_actions"])
    ap.add_argument("--provider", default=pol_cfg["llm"].get("provider", "lmstudio"))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    cache = resolve(data_cfg["cache_dir"])
    if not (cache / f"{args.ticker}.csv").exists():
        print(f"[FAIL] no data at {cache / (args.ticker + '.csv')}")
        print("Run: python scripts\\fetch_data.py")
        return 1

    bars = YahooCsvProvider(args.ticker, "", None, cache).load()
    segment = load_segment(bars, args.split, frac["train"], frac["val"], frac["test"])
    env = build_env(segment, f"{args.split}/{args.ticker}")

    if args.show_prompt:
        # prompt preview needs no server, so this runs before contacting LM Studio
        obs = env.reset()
        ctx = PolicyContext(env.builder, env.warmup)
        print(build_user_message(obs, ctx.date, ctx.price, args.actions))
        print(f"\n(first actionable bar: {segment[env.warmup].date}, "
              f"state length {len(obs)}, {env.last_index - env.warmup} decisions per episode)")
        return 0

    try:
        prov_cfg = resolve_provider(
            args.provider,
            pol_cfg["llm"],
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

    total_bars = env.last_index - env.warmup
    print(f"provider : {args.provider}")
    print(f"model    : {model}")
    print(f"data     : {args.ticker} {args.split} n={len(segment)} "
          f"{segment[0].date} -> {segment[-1].date}")
    print(f"decisions: {min(total_bars, args.max_steps or total_bars)} per episode "
          f"(1 LLM call each)")
    print(f"episodes : {args.episodes}\n")

    summary = []
    for ep in range(args.episodes):
        policy = LLMPolicy(
            client, n_actions=args.actions, system_prompt=pol_cfg["policy"]["system_prompt"]
        )
        rows: list[dict] = []
        env_for_cb = env

        def on_step(row, _e=env_for_cb):
            rows.append(row)
            if len(rows) % 25 == 0:
                print(f"    step {len(rows):>4}/{total_bars}  equity={_e.equity:.4f}")

        started = time.time()
        m = run_policy(env, policy, on_step=on_step, max_steps=args.max_steps)
        elapsed = time.time() - started

        out_path = resolve(args.out) / f"ep_{args.split}_{ep:02d}.csv"
        write_records(out_path, rows)

        rate = len(rows) / elapsed if elapsed else 0.0
        print(f"\n[{ep + 1}/{args.episodes}] {m['steps']} steps in {elapsed:.1f}s "
              f"({rate:.2f} steps/s)")
        print(f"  {env.format_metrics()}")
        print(f"  llm calls={policy.calls} parsed={policy.parsed} "
              f"parse_rate={policy.parse_rate * 100:.1f}%")
        hist = m["action_hist"]
        if hist:
            dist = "  ".join(
                f"{label_for(args.actions, k)}:{v}" for k, v in sorted(hist.items())
            )
            print(f"  actions: {dist}")
        if m["truncated"]:
            print(f"  (truncated at --max-steps {args.max_steps}, metrics are partial)")
        print(f"  record: {out_path}")
        summary.append(m)

    sharpes = [s["sharpe"] for s in summary if s["sharpe"] == s["sharpe"]]
    if sharpes:
        print(f"\navg over {len(sharpes)} episodes: sharpe={sum(sharpes) / len(sharpes):.2f}")
    print("compare against: python scripts\\run_baseline.py --split " + args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())