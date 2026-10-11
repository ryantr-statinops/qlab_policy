"""Reference numbers. Run this BEFORE trusting any LLM result.

    python scripts\run_baseline.py
    python scripts\run_baseline.py --split test
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from qlab_rl.config import load_data, load_env, resolve
from qlab_rl.data.provider import YahooCsvProvider
from qlab_rl.policies import rule
from qlab_rl.runner import build_env, load_segment, run_policy

SPLITS = ["train", "val", "test"]


def main() -> int:
    data_cfg = load_data()
    env_cfg = load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=data_cfg["ticker"])
    ap.add_argument("--split", choices=SPLITS + ["all"], default="all")
    ap.add_argument("--policies", default=",".join(rule.ALL_NAMES))
    ap.add_argument(
        "--lambda", dest="lambda_", type=float, default=env_cfg["env"]["lambda_risk"],
        help="risk aversion in reward = net - lambda * vol * |exposure|",
    )
    args = ap.parse_args()

    cache = resolve(data_cfg["cache_dir"])
    if not (cache / f"{args.ticker}.csv").exists():
        print(f"[FAIL] no data at {cache / (args.ticker + '.csv')}")
        print("Run: python scripts\\fetch_data.py")
        return 1

    bars = YahooCsvProvider(args.ticker, "", None, cache).load()
    splits = SPLITS if args.split == "all" else [args.split]
    names = [n.strip() for n in args.policies.split(",") if n.strip()]
    frac = env_cfg["split"]

    print(f"ticker {args.ticker}  bars={len(bars)}  "
          f"{bars[0].date} -> {bars[-1].date}  lambda={args.lambda_}\n")

    for split in splits:
        segment = load_segment(bars, split, frac["train"], frac["val"], frac["test"])
        env = build_env(segment, f"{split}/-{args.ticker}", {"lambda_risk": args.lambda_})
        span = f"{segment[0].date} -> {segment[-1].date}"
        print(f"--- {split}  n={len(segment)}  {span}")
        for name in names:
            policy = rule.make(name)
            m = run_policy(env, policy)
            print(f"  {name:<18} ret={m['total_return'] * 100:7.2f}%  "
                  f"sharpe={m['sharpe']:6.2f}  maxDD={m['max_drawdown'] * 100:7.2f}%  "
                  f"PF={m['profit_factor']:5.2f}  flat={m['flat_ratio'] * 100:5.1f}%")
        print()

    print("LLM policy has to beat these numbers. buy_hold is the one that matters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())