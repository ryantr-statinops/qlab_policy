# qlab_rl

LLM as an RL policy. A model hosted in LM Studio reads a market state and
picks a position exposure; a trading environment pays out a reward; everything
runs locally on one machine.

**Zero dependencies.** Python 3.11 stdlib only - `urllib`, `csv`, `tomllib`,
`statistics`. No `uv sync`, no `pip install`, no numpy.

## How the loop works

At each bar the model sees a 6-number state and answers with one exposure:

```
Date: 2022-02-22   Close: 429.57

Market state:
  return_1d            -1.07%
  return_5d            -2.15%
  return_20d           -2.33%
  vol_20d              +21.59%
  drawdown_20d         -6.11%
  exposure             +0.00

Choose exactly one position exposure:
  SHORT_ALL    = -1.00 (100% short)
  SHORT_HALF   = -0.50 (50% short)
  FLAT         = +0.00 (0% flat)
  LONG_HALF    = +0.50 (50% long)
  LONG_ALL     = +1.00 (100% long)

Reply in exactly this format and nothing else:
ACTION: <one of the option names>
REASON: <one short sentence>
```

Accounting, per bar:

```
cost    = fee_rate * |exposure - exposure_prev|
net_t   = exposure * (close[i+1]/close[i] - 1) - cost_t
reward_t = net_t - lambda * vol_20d(i) * |exposure|
equity *= (1 + net_t)
```

Shorting is symmetric: a falling market with a short earns money.

---

## Setup: LM Studio

LM Studio 0.4.19 is already installed at
`C:\Users\Admin\AppData\Local\Programs\LM Studio\LM Studio.exe`.

**There is no chat model on disk yet** - only an 84 MB embedding model, which
cannot answer prompts. Download one first.

GUI route:

1. Open LM Studio
2. **My Models** -> search `Qwen3 4B Instruct` -> Download (Q4_K_M, ~2.5 GB, fits 4 GB VRAM)
3. Load the model
4. **Developer** tab -> **Start Server** (default port 1234)

CLI route:

```powershell
lms pull qwen3-4b-instruct-q4_k_m
lms load qwen3-4b-instruct --context-length 8192
lms server start
lms ps            # confirm what is loaded
```

---

## Run it

```powershell
cd D:\01_Workspace\02_Finance_Trading\qlab_rl

python scripts\check_lmstudio.py      # is the server up, is a chat model loaded
python scripts\ask_model.py --repeat 3   # one decision, printed
python scripts\fetch_data.py          # SPY daily bars -> data/raw/SPY.csv
python -m unittest discover tests     # 57 tests
python scripts\run_baseline.py        # reference numbers on all three splits
python scripts\run_llm.py --episodes 1
python scripts\eval.py --split test
```

Useful flags:

| Flag | Effect |
|---|---|
| `run_llm.py --show-prompt` | print the exact prompt, no server needed |
| `run_llm.py --max-steps 40` | short run to measure throughput |
| `run_llm.py --actions 3` | long-only, drops the short options |
| `ask_model.py --free` | open-ended question, no option list |
| `eval.py --skip-llm` | baselines only, no server needed |

---

## What to look at

**1. `parse_rate`.** How often the model returns a legal action. Below 80% the
prompt is not working and no performance number means anything. The raw text of
every call is written to `outputs/ep_*.csv`.

**2. `actions:` distribution.** If one action gets 95% of the bars, the model
is not reading the state. Set `--actions 3` to give it less room to hide.

**3. Stability.** `ask_model.py --repeat 3` should give the same answer three
times at temperature 0. If it does not, every later measurement is noise.

**4. The bar to clear.** From `run_baseline.py` on SPY test split:

| policy | sharpe | ret | maxDD |
|---|---|---|---|
| buy_hold | **1.02** | +41.5% | -19.0% |
| fixed_half | **1.02** | +19.9% | -9.9% |
| sma20_long_flat | 0.20 | +3.2% | -11.8% |
| random | -1.71 | -36.2% | -41.2% |

A 4B model reading six numbers is not going to beat `buy_hold` on SPY. That is
known in advance and is not a bug. Success for v1 means the harness runs,
numbers are reproducible, and the logs are readable - not that it beats the
market. Where an LLM policy *could* genuinely win is context a numeric policy
cannot consume: news text, research notes, chart descriptions. That is the next
thing to build, not weight training.

---

## Config

`configs/env.toml` - fee (10 bps), `lambda_risk`, warmup, the 60/20/20 split,
state lookbacks.
`configs/policy.toml` - endpoint, model, temperature, `n_actions`, system prompt.
`configs/data.toml` - ticker, date range, cache directory.

Split is 60/20/20 by bar count, chronological: train 2015-01 -> 2022-01,
val 2022-01 -> 2024-05, test 2024-05 -> 2026-10.

Swap `n_actions = 5` for `3` in `configs/policy.toml` to restrict the model to
long-only. That is the easiest way to test whether shorting is what is hurting.

## Layout

```
configs/          data.toml, env.toml, policy.toml
scripts/          check_lmstudio, ask_model, fetch_data,
                  run_baseline, run_llm, eval
src/qlab_rl/
  config.py       tomllib loader
  runner.py       drives a policy through an env, writes records
  llm/client.py   LM Studio OpenAI-compatible client
  data/provider.py Yahoo Finance OHLCV + CSV cache
  env/state.py    observation builder (no lookahead)
  env/trading_env.py  step / equity / reward / metrics
  env/splits.py   60/20/20 chronological
  policies/       base, actions, rule, llm
tests/            57 tests, no LM Studio required
data/raw/         cached CSV (gitignored)
outputs/          per-episode records (gitignored)
```

## Known limits

- Single asset, daily bars, no corporate actions, no borrow fee on shorts.
- One LLM call per bar, strictly sequential - roughly 500-1,400 calls per
  episode. At ~1 s/call that is 10-25 minutes per episode.
- No weight training. 4 GB VRAM is not enough for GRPO/PPO on a 4B model.