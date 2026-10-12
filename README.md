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
limits.

## Bar to clear

SPY test split, from `run_baseline.py`: `buy_hold` sharpe **1.02**,
ret +41.5%, maxDD -19.0%. A 4B model reading six numbers is not going to
beat it — known in advance, not a bug. v1 success = harness runs, numbers
are reproducible, logs are readable. See
[docs/14-outputs-eval.md](docs/14-outputs-eval.md).
