"""Translating between key codes and characters on the current keyboard layout.

A chord has to mean the same thing however the keyboard is laid out, so its identity comes
from the letter a key produces with no modifiers held, not from the character the
application would have received. Pressing Shift-T is still the letter "t" as far as the
chord is concerned; the Shift only decides capitalisation later.

The mapping is obtained by asking the system what a synthetic keystroke would produce.
That is simpler than translating through the Text Input Services tables by hand, and it
follows the layout the user actually has selected.

It only works in one direction. An event's Unicode string is computed when the event is
created and setting modifier flags afterwards does not recompute it, so there is no way
to ask "which key gives a capital T" this way. That does not matter, because text is
emitted by overriding an event's Unicode string outright rather than by pressing the keys
that would have produced it, which is layout-independent and needs no reverse table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import Quartz

# Key codes that are the same on every layout, since they are physical positions rather
# than characters.
SPACE = 49
RETURN = 36
TAB = 48
BACKSPACE = 51
ESCAPE = 53
DELETE = 117
ARROWS = frozenset({123, 124, 125, 126})  # left, right, down, up
HOME_END_PAGE = frozenset({115, 116, 119, 121})
RIGHT_OPTION = 61
RIGHT_COMMAND = 54

# Keys that move the caret somewhere our own record of the text cannot follow.
NAVIGATION = ARROWS | HOME_END_PAGE | {RETURN, TAB, ESCAPE}

MAX_KEYCODE = 128


@dataclass
class KeyboardLayout:
    """Key code to character, for the layout in effect when this was built."""

    to_char: dict[int, str] = field(default_factory=dict)
    to_keycode: dict[str, int] = field(default_factory=dict)

    @classmethod
    def current(cls) -> KeyboardLayout:
        layout = cls()
        layout.refresh()
        return layout

    def refresh(self) -> None:
        """Rebuild the tables. Call when the selected input source changes."""
        self.to_char.clear()
        self.to_keycode.clear()
        for keycode in range(MAX_KEYCODE):
            character = _character_for(keycode)
            if character:
                self.to_char[keycode] = character
                self.to_keycode.setdefault(character, keycode)

    def letter(self, keycode: int) -> str | None:
        """The a-z letter this key produces unmodified, or None if it is not a letter."""
        character = self.to_char.get(keycode)
        if character and len(character) == 1 and character.isalpha() and character.isascii():
            return character.lower()
        return None

    def letter_keycode(self, letter: str) -> int | None:
        """The key that produces this letter, for re-sending a letter we held back."""
        return self.to_keycode.get(letter.lower())


def _character_for(keycode: int) -> str | None:
    """Ask the system what this key types with no modifiers held."""
    event = Quartz.CGEventCreateKeyboardEvent(None, keycode, True)
    if event is None:
        return None
    count, text = Quartz.CGEventKeyboardGetUnicodeString(event, 8, None, None)
    if not count or not text:
        return None
    # Control characters (return, tab, escape, backspace) are dealt with by key code, and
    # dead keys for accents produce nothing useful on their own.
    if len(text) != 1 or ord(text) < 0x20 or ord(text) == 0x7F:
        return None
    return text
