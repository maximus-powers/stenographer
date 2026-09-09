"""The predictor interface shared by every ranking strategy.

Everything that turns a chord into ranked words implements this: the frequency baseline,
the trained model, and the socket client the tap process talks to. Keeping the surface
this small is what lets the harness score them all with the same code, and what lets the
brain be swapped for an in-process predictor while debugging.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Request:
    """A chord plus everything known about where it was typed."""

    letters: str  # sorted distinct chord letters
    press_order: str = ""  # order the keys went down, a weak signal at best
    left: Sequence[str] = field(default_factory=tuple)  # preceding tokens, nearest last
    right: Sequence[str] = field(default_factory=tuple)  # following tokens, nearest first
    sentence_start: bool = False
    shift: bool = False
    caps_lock: bool = False


@runtime_checkable
class Predictor(Protocol):
    """Ranks the words a chord might have meant, best first."""

    def predict(self, request: Request, k: int = 5) -> list[str]:
        ...

    def warmup(self) -> None:
        ...
