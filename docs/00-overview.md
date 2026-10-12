# 00 — Overview

`qlab_rl`: LLM as an RL policy for trading. A model hosted in LM Studio reads
a market state and picks a position exposure; a trading environment pays out
a reward; everything runs locally on one machine.

Zero dependencies. Python 3.11 stdlib only — `urllib`, `csv`, `tomllib`,
`statistics`. No `uv sync`, no `pip install`, no numpy.

Loop, per bar:

```
market state (6 numbers) -> LLM -> exposure in [-1, 1] -> env -> reward + equity
```

Non-goal for v1: beating the market. A 4B model reading six numbers is not
going to beat `buy_hold` on SPY — known in advance, not a bug. Success for v1
means the harness runs, numbers are reproducible, and the logs are readable.

Where an LLM policy *could* genuinely win is context a numeric policy cannot
consume: news text, research notes, chart descriptions. That is the next thing
to build, not weight training.
