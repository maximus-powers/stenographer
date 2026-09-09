"""Sending keystrokes to whichever application has focus.

Text is emitted by creating a key event and overriding its Unicode string, rather than by
pressing the keys that would have produced the text. That is layout-independent, needs no
reverse key table, and is how other macOS text expanders do it.

Two details are not optional. Every event we post carries a marker in its source user
data, because `CGEventPost` delivers ahead of event taps and our own output would
otherwise come straight back into the detector as if the user had typed it. And each
event carries at most twenty UTF-16 units, which is where the system truncates.

The pasteboard with a synthetic Command-V was considered and rejected. It would overwrite
whatever the user had copied, Command-V is bound to something else in terminals and vim,
it can carry styled text into a plain field, and it saves nothing at the length of a
single word.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import Quartz

from stenographer.keyboard.layout import BACKSPACE

# Stamped into the user data of everything we post, so the tap can recognise its own
# output. Any value distinct from zero works; this one is just recognisable in a log.
MARKER = 0x53544E30  # "STN0"

# The system truncates a keyboard event's Unicode string at about this many UTF-16 units.
MAX_UNITS = 20


def is_ours(event) -> bool:
    """Did we post this event? Called first in the tap callback, so keep it cheap."""
    return (
        Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData) == MARKER
    )


def _chunks(text: str) -> list[str]:
    """Split text so no chunk exceeds the Unicode string limit.

    The limit counts UTF-16 units, not characters, so anything outside the basic plane
    (an emoji, say) counts double and a surrogate pair must not be split down the middle.
    """
    chunks: list[str] = []
    current: list[str] = []
    units = 0
    for character in text:
        width = 2 if ord(character) > 0xFFFF else 1
        if units + width > MAX_UNITS:
            chunks.append("".join(current))
            current, units = [], 0
        current.append(character)
        units += width
    if current:
        chunks.append("".join(current))
    return chunks


@dataclass
class Emitter:
    """Posts synthetic keystrokes, marked as ours."""

    # Pause between events. Zero is right for native applications; a millisecond or two
    # helps if an Electron app is seen to drop characters.
    gap_seconds: float = 0.0
    dry_run: bool = False
    log: list[str] | None = None

    def __post_init__(self) -> None:
        self.source = None if self.dry_run else Quartz.CGEventSourceCreate(
            Quartz.kCGEventSourceStatePrivate
        )

    # -- primitives -----------------------------------------------------------

    def _post(self, event) -> None:
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGEventSourceUserData, MARKER)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        if self.gap_seconds:
            time.sleep(self.gap_seconds)

    def _record(self, description: str) -> None:
        if self.log is not None:
            self.log.append(description)

    # -- text -----------------------------------------------------------------

    def type_text(self, text: str) -> None:
        """Insert `text` at the caret."""
        if not text:
            return
        self._record(f"type {text!r}")
        if self.dry_run:
            return
        for chunk in _chunks(text):
            for pressed in (True, False):
                event = Quartz.CGEventCreateKeyboardEvent(self.source, 0, pressed)
                Quartz.CGEventKeyboardSetUnicodeString(event, len(chunk), chunk)
                self._post(event)

    def backspace(self, count: int) -> None:
        """Delete `count` characters before the caret."""
        if count <= 0:
            return
        self._record(f"backspace {count}")
        if self.dry_run:
            return
        for _ in range(count):
            for pressed in (True, False):
                event = Quartz.CGEventCreateKeyboardEvent(self.source, BACKSPACE, pressed)
                self._post(event)

    def key(self, keycode: int, *, shift: bool = False) -> None:
        """Press and release one key by code."""
        self._record(f"key {keycode}{' +shift' if shift else ''}")
        if self.dry_run:
            return
        flags = Quartz.kCGEventFlagMaskShift if shift else 0
        for pressed in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(self.source, keycode, pressed)
            Quartz.CGEventSetFlags(event, flags)
            self._post(event)

    def repost(self, event, *, synthetic_up: bool = True) -> None:
        """Send back a keystroke we held on to, exactly as it arrived.

        Copying the original preserves its key code, modifier flags and Unicode string, so
        a letter that was held back and then released reaches the application indistinct
        from one that was never intercepted.
        """
        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        self._record(f"repost keycode {keycode}")
        if self.dry_run:
            return
        copy = Quartz.CGEventCreateCopy(event)
        self._post(copy)
        if synthetic_up:
            release = Quartz.CGEventCreateKeyboardEvent(self.source, int(keycode), False)
            Quartz.CGEventSetFlags(release, Quartz.CGEventGetFlags(event))
            self._post(release)
