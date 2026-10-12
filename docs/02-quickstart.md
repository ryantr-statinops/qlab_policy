# 02 — Quickstart

```powershell
cd D:\01_Workspace\02_Finance_Trading\qlab_rl

python scripts\check_lmstudio.py      # is the server up, is a chat model loaded
python scripts\ask_model.py --repeat 3   # one decision, printed
python scripts\fetch_data.py          # SPY daily bars -> data/raw/SPY.csv
python -m unittest discover tests     # 57 tests, no LM Studio required
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

`ask_model.py --repeat 3` is the smoke test: same answer three times at
temperature 0 means the stack is stable. If it varies, every later
measurement is noise.
