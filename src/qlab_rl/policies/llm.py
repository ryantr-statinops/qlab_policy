"""LLM policy: the language model is the decision maker.

One chat completion per bar. The model reads a compact state table and answers
with a single exposure choice, e.g. `ACTION: LONG_HALF`.

Design notes
------------
* temperature 0 by default, because a policy that flips its mind between runs
  makes every downstream metric meaningless.
* The prompt is deliberately short. On a 4 GB card the model is partly on CPU;
  every extra token costs wall-clock on every single bar.
* Raw values are shown as percentages rather than z-scores. A 4B model reads
  "+1.20%" more reliably than it reads "0.83 sigma".
"""

from __future__ import annotations

import re
from typing import Sequence

from .actions import action_set, labels, nearest, value_of

EXPOSURE_FEATURE = "exposure"


def feature_names(
    ret_windows: Sequence[int] = (1, 5, 20),
    vol_window: int = 20,
    drawdown_window: int = 20,
) -> list[str]:
    """Mirror of StateBuilder.feature_names, for prompt building."""
    return (
        [f"return_{w}d" for w in ret_windows]
        + [f"vol_{vol_window}d", f"drawdown_{drawdown_window}d", EXPOSURE_FEATURE]
    )


def format_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def build_user_message(
    obs: Sequence[float],
    date: str,
    price: float,
    n_actions: int,
    memories: Sequence[dict] | None = None,
    names: Sequence[str] | None = None,
) -> str:
    names = list(names or feature_names())
    if len(names) != len(obs):
        raise ValueError(f"obs has {len(obs)} values but {len(names)} feature names")

    rows = []
    for label, value in zip(names, obs):
        if label == EXPOSURE_FEATURE:
            rows.append(f"  {label:<20} {value:+.2f}")
        else:
            rows.append(f"  {label:<20} {format_pct(value)}")

    options = "\n".join(
        f"  {name:<12} = {value:+.2f} "
        f"({abs(value) * 100:.0f}% "
        f"{'short' if value < 0 else 'long' if value > 0 else 'flat'})"
        for name, value in action_set(n_actions)
    )

    parts: list[str] = []
    if memories:
        parts.append("Decisions that worked well in earlier sessions:")
        for m in memories:
            parts.append(f"  {m['state']}  ->  {m['action']}  ({m['outcome']})")
        parts.append("")

    parts.append(f"Date: {date}   Close: {price:.2f}")
    parts.append("")
    parts.append("Market state:")
    parts.extend(rows)
    parts.append("")
    parts.append("Choose exactly one position exposure:")
    parts.append(options)
    parts.append("")
    parts.append("Reply in exactly this format and nothing else:")
    parts.append("ACTION: <one of the option names>")
    parts.append("REASON: <one short sentence>")
    return "\n".join(parts)


_ACTION_RE = re.compile(r"ACTION\s*[:#\-]?\s*([A-Z][A-Z_]{2,})", re.IGNORECASE)


def parse_action(text: str, n_actions: int) -> tuple[float | None, str]:
    """Return (exposure, matched_label). exposure is None when nothing matched.

    Small models often ignore the requested format and just name the action in
    prose, so a bare-label scan backs up the explicit ACTION: match.
    """
    legal = labels(n_actions)
    match = _ACTION_RE.search(text or "")
    if match:
        label = match.group(1).upper()
        if label in legal:
            return value_of(n_actions, label), label
    upper = (text or "").upper()
    for label in legal:
        if re.search(rf"\b{label}\b", upper):
            return value_of(n_actions, label), label
    return None, "UNPARSED"


class LLMPolicy:
    name = "llm"

    def __init__(
        self,
        client,
        n_actions: int = 5,
        system_prompt: str | None = None,
        memories: Sequence[dict] | None = None,
        fallback: float | None = None,
    ) -> None:
        self.client = client
        self.n_actions = n_actions
        self.system_prompt = system_prompt or (
            "You are a disciplined swing-trading risk manager. You are given a market "
            "state and you pick one position exposure. Answer in the exact format asked."
        )
        self.memories = list(memories or [])
        self.fallback = nearest(n_actions, 0.0) if fallback is None else fallback
        self.calls = 0
        self.parsed = 0
        self.last_parse_ok = False
        self.responses: list[str] = []

    def reset(self) -> None:
        self.calls = 0
        self.parsed = 0
        self.last_parse_ok = False
        self.responses = []

    @property
    def parse_rate(self) -> float:
        return self.parsed / self.calls if self.calls else float("nan")

    def act(self, obs: Sequence[float], ctx) -> float:
        user = build_user_message(
            obs,
            ctx.date,
            ctx.price,
            self.n_actions,
            memories=self.memories,
            names=getattr(ctx.builder, "feature_names", None),
        )
        try:
            response = self.client.chat(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user},
                ]
            )
            text = self.client.text_of(response)
        except Exception as exc:  # network / server hiccup must not kill the episode
            self.calls += 1
            self.last_parse_ok = False
            self.responses.append(f"<ERROR> {exc}")
            return self.fallback

        self.calls += 1
        self.responses.append(text)
        exposure, _label = parse_action(text, self.n_actions)
        self.last_parse_ok = exposure is not None
        if exposure is None:
            return self.fallback
        self.parsed += 1
        return exposure

    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "parsed": self.parsed,
            "parse_rate": self.parse_rate,
        }