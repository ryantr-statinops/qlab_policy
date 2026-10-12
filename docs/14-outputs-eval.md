# 14 — Outputs & eval

Sources: `src/qlab_rl/runner.py`, `scripts/eval.py`, `scripts/run_llm.py`.

Per-episode records: `outputs/ep_*.csv` (gitignored), written by
`write_records`. Columns (`RECORD_FIELDS`):

```
step, date, price, action, reward, equity,
gross_return, cost, turnover, response, parse_ok
```

`response` is the raw model text truncated to 400 chars with newlines
flattened. For rule policies `parse_ok` is 1 unless the text looks like an
`<ERROR>`; for LLM policies it is `last_parse_ok`.

What to look at:

1. **`parse_rate`.** How often the model returns a legal action. Below 80%
   the prompt is not working and no performance number means anything.
2. **`actions:` distribution.** If one action gets 95% of the bars, the model
   is not reading the state. Set `--actions 3` to give it less room to hide.
3. **Stability.** `ask_model.py --repeat 3` should give the same answer three
   times at temperature 0. If not, every later measurement is noise.
4. **The bar to clear.** `eval.py` ranks LLM vs every rule policy by sharpe
   on the chosen split; `eval.py --skip-llm` runs baselines only.
