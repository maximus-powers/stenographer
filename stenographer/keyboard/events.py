"""Key events, independent of macOS.

The detector is the piece most likely to get the subtle cases wrong, so it is kept free
of pyobjc: everything it sees is one of these plain records. That is what lets the tests
drive it with hand-written timings instead of a real keyboard.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz")


class Kind(Enum):
    DOWN = "down"
    UP = "up"
    FLAGS = "flags"  # a modifier changed on its own


@dataclass(frozen=True)
class KeyEvent:
    """One keystroke as the tap saw it.

    `key` is the layout-independent base letter, resolved by translating the keycode with
    no modifiers applied, so a chord means the same thing whatever Shift or Option did to
    the character the application would have received.
    """

    kind: Kind
    t_ms: float
    key: str | None = None
    keycode: int = -1
    shift: bool = False
    caps_lock: bool = False
    autorepeat: bool = False
    command: bool = False  # Command, Control or Option held: a shortcut, never a chord
    # The hold-to-chord key is down. The tap decides which physical key that is and sets
    # this on every event, so the detector never has to know about keycodes or flag masks.
    chord_modifier: bool = False

    @property
    def is_letter(self) -> bool:
        return self.key in LETTERS and not self.command

    @property
    def is_down(self) -> bool:
        return self.kind is Kind.DOWN

    @property
    def is_up(self) -> bool:
        return self.kind is Kind.UP

    def at(self, t_ms: float) -> KeyEvent:
        return replace(self, t_ms=t_ms)


def down(key: str, t_ms: float, **kwargs) -> KeyEvent:
    return KeyEvent(kind=Kind.DOWN, t_ms=t_ms, key=key, **kwargs)


def up(key: str, t_ms: float, **kwargs) -> KeyEvent:
    return KeyEvent(kind=Kind.UP, t_ms=t_ms, key=key, **kwargs)


def other(t_ms: float, keycode: int = 49, **kwargs) -> KeyEvent:
    """A non-letter keystroke, such as space, punctuation or Enter."""
    return KeyEvent(kind=Kind.DOWN, t_ms=t_ms, key=None, keycode=keycode, **kwargs)
