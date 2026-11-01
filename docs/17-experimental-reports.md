# 17 — Experimental reports: v1 local test (quyet-1.0-medium)

Source records: `outputs/ep_val_00.csv`, `outputs/ep_test_00.csv`.

Empirical findings from live runs of the zero-dependency RL trading harness using an offline 4B instruction-tuned LLM (`quyet-1.0-medium`, Qwen3.5 architecture, 3.3 GB GGUF) served by LM Studio.

---

## 1. Environment & Hardware Setup

- **Host GPU:** NVIDIA Quadro T1000 (4,096 MiB VRAM).
- **VRAM Contention:** Host background applications shared VRAM, requiring LM Studio to split layers between GPU and CPU RAM.
- **Latency / Throughput:** Cold prompt probe achieved ~2.84s/call; sustained sequential execution in live episodes averaged ~7.2s – 7.6s/call.
- **Inference Configuration:**
  - `temperature = 0.0`
  - `max_tokens = 160`
  - `reasoning_effort = "none"` (verified necessary via `scripts/probe_latency.py` to prevent reasoning models consuming the token budget before emitting text).
- **Format Integrity:** 100% legal action parsing (75/75 total decisions across `val` and `test` splits) via strict `ACTION:` parser and fallback word-boundary regex scan.

---

## 2. Validation Run (25 bars: 2022-02-23 -> 2022-03-29)

Data segment: SPY at the outbreak of the Russia-Ukraine conflict (initial drop followed by a +7.3% rebound from $421 to $461).

### Benchmark Comparison (25 bars)

| Policy | Return | Sharpe | Max Drawdown | Action / Exposure |
|---|---|---|---|---|
| `buy_hold` | **+7.34%** | **+3.13** | -4.94% | 100% Long (+1.00) |
| `sma20_long_flat` | **+3.76%** | **+4.45** | **-1.29%** | Flat 76%, Long 24% |
| `random` | +7.47% | +4.49 | -2.25% | Random actions |
| **`quyet-1.0-medium` (LLM)** | **-3.77%** | **-3.24** | **-5.14%** | **100% SHORT_HALF (-0.50)** |

### Diagnostic Analysis (`outputs/ep_val_00.csv`)

1. **Action Collapse:** The model selected `SHORT_HALF` for 25 out of 25 consecutive bars.
2. **Elevated Volatility Bias:** The observation state displayed `vol_20d > 20%` and negative returns. The model anchored entirely on risk aversion, continually shorting into an advancing market.
3. **Exposure Misinterpretation:** At step 21, the model stated:
   > `"SHORT_HALF -50% short to hedge 50% long exposure"`
   Although the actual portfolio position was already -0.50 short, the model hallucinated that it held a long position requiring hedging.

---

## 3. Test Split Run (50 bars: 2024-07-03 -> 2024-09-12)

Data segment: SPY during Summer 2024, containing the Yen carry trade unwind crash and subsequent V-shaped recovery.

### Benchmark Comparison (50 bars)

| Policy | Return | Sharpe | Max Drawdown | Action / Exposure |
|---|---|---|---|---|
| `sma20_long_flat` | **+1.19%** | **+0.82** | **-2.59%** | Flat 68%, Long 32% |
| `buy_hold` | **+1.73%** | **+0.59** | -8.41% | 100% Long (+1.00) |
| `fixed_half` | +0.94% | +0.59 | -4.26% | 50% Long (+0.50) |
| `sma20_cross` | -1.14% | -0.32 | -5.38% | Flat 38%, Active swing |
| **`quyet-1.0-medium` (LLM)** | **-4.96%** | **-3.32** | **-6.03%** | **SHORT_HALF: 56%, FLAT: 24%, LONG_HALF: 20%** |
| `random` | -8.16% | -3.30 | -8.75% | Random actions |

### Key Observations & Case Studies (`outputs/ep_test_00.csv`)

1. **Dynamic Switching:** Unlike the validation split, the model demonstrated multi-state transitions:
   - `FLAT`: 12 bars (24%)
   - `LONG_HALF`: 10 bars (20%)
   - `SHORT_HALF`: 28 bars (56%)
2. **Crash Anticipation (Step 22, 2024-08-05):**
   - On the August 5 "Black Monday" sell-off, SPY dropped sharply to $517.38.
   - The model executed `ACTION: SHORT_HALF` with reason `"Protect capital from continued downside"`.
   - Result: Booked a **+1.45% gross return** on the day, moving equity up from `0.9752` to `0.9889`.
3. **The V-Shaped Rebound Trap (Steps 23–36):**
   - After the plunge, SPY staged a rapid recovery from $517 to $560 (+8.3%).
   - The model remained locked in `SHORT_HALF` across 14 trading days because trailing `vol_20d` remained elevated (> 16-20%).
   - The persistent short position degraded equity from `0.9889` down to `0.9482`.
4. **Reasoning Inconsistencies:**
   - At Step 49, the model provided contradictory rationale: `"SHORT_HALF -50% short protects downside while capturing upside"`, reflecting semantic drift under numeric-only state input.

---

## 4. Engineering Improvements Delivered

- **Incremental Record Flushing (`scripts/run_llm.py`):**
  Added periodic disk flush of episode rows every 25 steps and inside a `finally:` block, preventing data loss during long-running background executions.
- **Cross-Platform Path Resolution (`tests/test_config.py`):**
  Resolved Windows path root handling for `resolve()`, bringing the test suite to **117/117 passing tests**.

---

## 5. Next Steps

1. **Clarify Prompt Exposure Semantics:** Explicitly define `exposure` as current net portfolio position (negative = currently short, positive = currently long, zero = flat).
2. **Evaluate Restricted Action Space (`--actions 3`):** Benchmark with `FLAT`, `LONG_HALF`, `LONG_ALL` only, removing shorting to isolate trend-following capability.
3. **In-Context Learning / Reflection Memory (P2):** Inject memory prompts derived from empirical failure modes (e.g. recognizing post-drawdown recovery vs sustained downtrend).
