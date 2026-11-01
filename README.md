# qlab_rl

LLM as an RL policy. A model hosted in LM Studio reads a market state and
picks a position exposure; a trading environment pays out a reward; everything
runs locally on one machine.

**Zero dependencies.** Python 3.11 stdlib only - `urllib`, `csv`, `tomllib`,
`statistics`. No `uv sync`, no `pip install`, no numpy.

Loop, per bar: 6-number state -> LLM -> exposure in `[-1, 1]` -> env ->
reward + equity. See [docs/00-overview.md](docs/00-overview.md) and
[docs/07-trading-env.md](docs/07-trading-env.md).

## Run it

```powershell
python scripts\check_lmstudio.py      # is the server up, is a chat model loaded
python scripts\ask_model.py --repeat 3   # one decision, printed
python scripts\fetch_data.py          # SPY daily bars -> data/raw/SPY.csv
python -m unittest discover tests     # 57 tests, no LM Studio required
python scripts\run_baseline.py        # reference numbers on all three splits
python scripts\run_llm.py --episodes 1
python scripts\eval.py --split test
```

Setup detail: [docs/01-setup.md](docs/01-setup.md). Flags and smoke test:
[docs/02-quickstart.md](docs/02-quickstart.md).

## Docs

Full notes in [docs/](docs/README.md): architecture, data, splits, state,
env math, metrics, actions, baselines, LLM policy/client, config, outputs,
limits, providers.

Any OpenAI-compatible endpoint works — local (LM Studio, Ollama, vLLM) or
cloud (OpenRouter, OpenAI) via `--provider`. See
[docs/16-providers.md](docs/16-providers.md).

## Bar to clear

SPY test split, from `run_baseline.py`: `buy_hold` sharpe **1.02**,
ret +41.5%, maxDD -19.0%. A 4B model reading six numbers is not going to
beat it — known in advance, not a bug. v1 success = harness runs, numbers
are reproducible, logs are readable. See
[docs/14-outputs-eval.md](docs/14-outputs-eval.md).

## Initial Experimental Run (v1 Smoke Test)

Executed on local LM Studio using `quyet-1.0-medium` (4B, Qwen3.5 arch, 3.3 GB) on the first 25 bars of the `val` split (2022-02-23 -> 2022-03-29):

### Quantitative Comparison (25 bars)

| Policy | Return | Sharpe | Max Drawdown | Action / Exposure |
|---|---|---|---|---|
| `buy_hold` | **+7.34%** | **+3.13** | -4.94% | 100% Long (+1.00) |
| `sma20_long_flat` | **+3.76%** | **+4.45** | **-1.29%** | Flat 76%, Long 24% |
| `random` | +7.47% | +4.49 | -2.25% | Random mix |
| `quyet-1.0-medium` (LLM) | **-3.77%** | **-3.24** | **-5.14%** | **100% SHORT_HALF (-0.50)** |

### Diagnostic Findings from `outputs/ep_val_00.csv`

1. **Parse rate & throughput:** 100% legal action parsing (25/25) with `reasoning_effort = "none"`, average latency ~2.8s/call.
2. **Action collapse:** Model picked `SHORT_HALF` for 25/25 steps. With elevated volatility (`vol_20d` > 20%) and negative trailing drawdown, the model persistently shorted into a +7.3% SPY rebound.
3. **Exposure hallucination:** At step 21, the model stated `SHORT_HALF -50% short to hedge 50% long exposure` while its prior position was already -0.50 short. The prompt must explicitly clarify `exposure` as current net portfolio position.

### Test Split Initial Run (50 bars: 2024-07-03 -> 2024-09-12)

Executed on `test` split (`outputs/ep_test_00.csv`):

| Policy | Return | Sharpe | Max Drawdown | Action / Exposure |
|---|---|---|---|---|
| `buy_hold` | **+1.73%** | **+0.59** | -8.41% | 100% Long (+1.00) |
| `sma20_long_flat` | **+1.19%** | **+0.82** | **-2.59%** | Flat 68%, Long 32% |
| `quyet-1.0-medium` (LLM) | **-4.96%** | **-3.32** | **-6.03%** | SHORT_HALF 56%, FLAT 24%, LONG_HALF 20% |
| `random` | -8.16% | -3.30 | -8.75% | Random mix |

**Key Observations**:
1. **Dynamic shifting:** Unlike `val`, the model actively switched between `FLAT` (24%), `LONG_HALF` (20%), and `SHORT_HALF` (56%).
2. **Crash timing vs rebound trap:** At step 22 (2024-08-05, Yen carry crash), the model correctly chose `SHORT_HALF`, booking a +1.45% profit. However, it stayed short throughout the subsequent V-shaped recovery to $560 due to lingering 20d volatility metrics.
3. **Reasoning contradictions:** At step 49, the model hallucinated that `SHORT_HALF -50% short protects downside while capturing upside`.
