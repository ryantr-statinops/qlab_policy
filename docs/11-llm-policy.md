# 11 — LLM policy

Source of truth: `src/qlab_rl/policies/llm.py`.

One chat completion per bar. The model reads a compact state table and
answers with a single exposure choice, e.g. `ACTION: LONG_HALF`.

Prompt: `Date / Close` header, one `label value` row per feature, the action
option list, then a strict `ACTION: / REASON:` format demand. Design notes
from the module docstring:

- temperature 0 by default — a policy that flips its mind between runs makes
  every downstream metric meaningless.
- Prompt deliberately short — on a 4 GB card the model is partly on CPU;
  every extra token costs wall-clock on every single bar.
- Raw values as percentages (`+1.20%`), not z-scores — a 4B model reads
  percentages more reliably.

Parsing (`parse_action`): explicit `ACTION:` regex first, then a bare-label
word-boundary scan as backup, because small models often name the action in
prose. Anything unmatched -> `UNPARSED` -> `fallback` (nearest legal to
0.0, i.e. FLAT). Network/server errors also fall back without killing the
episode.

`LLMPolicy` tracks `calls / parsed / parse_rate / last_parse_ok / responses`.
`build_user_message` accepts an optional `memories` block ("decisions that
worked well in earlier sessions") — hook only, no writer yet.
`ask_model.py` uses a fixed `DEMO_OBS` for its single-decision smoke test.
