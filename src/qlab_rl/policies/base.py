"""Policy interface."""

from __future__ import annotations

from typing import Protocol, Sequence

from ..env.state import StateBuilder


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, obs: Sequence[float], ctx: "PolicyContext") -> float: ...


class PolicyContext:
    """Read-only view handed to policies; keeps prompt building out of the env."""

    def __init__(self, builder: StateBuilder, index: int, episode: int = 0, total_episodes: int = 1) -> None:
        self.builder = builder
        self.index = index
        self.episode = episode
        self.total_episodes = total_episodes

    @property
    def date(self) -> str:
        return self.builder.bars[self.index].date

    @property
    def price(self) -> float:
        return self.builder.bars[self.index].close