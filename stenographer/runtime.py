"""Wiring the detector, the composer and the emitter into one keystroke handler.

The tap hands over one key event at a time and needs one answer: does this keystroke
reach the application or not. Everything between those two points lives here.

The awkward part is ordering. When a keystroke both closes a group and belongs to the
application, the held-back letters have to arrive first. Returning the event from the
callback sends it on its way immediately, while our own output is injected separately,
so the two can cross. The fix is to never do both at once: if anything was injected while
handling an event, that event is swallowed and re-injected afterwards, so everything
reaches the application through the same path and in the right order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from stenographer.keyboard.composer import Backspaces, TextComposer, TypeText
from stenographer.keyboard.detector import (
    ChordDetector,
    CommitChord,
    Consume,
    PassThrough,
    Retract,
    TypeLetters,
)
from stenographer.keyboard.events import KeyEvent
from stenographer.keyboard.layout import BACKSPACE, NAVIGATION, SPACE, TAB

# How many alternatives to keep for the cycle key.
CANDIDATES = 5


def _with_case(event: KeyEvent) -> str:
    """The character this keystroke should produce, honouring Shift and Caps Lock."""
    if not event.key:
        return ""
    # Shift and Caps Lock cancel each other out, as on any keyboard.
    return event.key.upper() if event.shift != event.caps_lock else event.key


@dataclass
class RuntimeStats:
    chords: int = 0
    letters: int = 0
    cycles: int = 0
    undos: int = 0
    no_candidates: int = 0


@dataclass
class Runtime:
    """Turns key events into decisions and injected text."""

    detector: ChordDetector
    composer: TextComposer
    emitter: object  # an Emitter, or anything with type_text/backspace/repost
    predict: Callable[[CommitChord], list[str]]
    # Keys that repair the last committed word. Both are only intercepted while a commit
    # is still live, so they behave normally the rest of the time.
    cycle_keycode: int | None = TAB
    undo_keycode: int | None = BACKSPACE
    # Needed to tell which printable character a non-letter key produces, so the composer
    # can act on punctuation. The tap owns the real one; without it only the keys that are
    # the same on every layout are recognised.
    layout: object | None = None
    on_text: Callable[[str], None] | None = None
    on_nav: Callable[[str], None] | None = None
    stats: RuntimeStats = field(default_factory=RuntimeStats)

    # -- the tap's question ---------------------------------------------------

    def handle(self, event: KeyEvent, raw=None) -> bool:
        """Return True to let the keystroke through, False to swallow it."""
        if self._try_repair(event) == "handled":
            return False
        # A repair key that did not apply is an ordinary keystroke, so it still goes
        # through the detector: Backspace and Tab both have to be able to close an open
        # group before they reach the application.
        actions = self.detector.feed(event)
        return self._run(actions, event, raw)

    def on_timer(self, t_ms: float) -> None:
        """Give the detector a chance to decide a group whose window has closed."""
        actions = self.detector.on_timer(t_ms)
        if actions:
            self._run(actions, None, None)

    def next_deadline(self) -> float | None:
        return self.detector.next_deadline()

    # -- acting on the detector's decisions -----------------------------------

    def _run(self, actions: list, event: KeyEvent | None, raw) -> bool:
        decision: bool | None = None
        decision_raw = None
        injected = False

        for action in actions:
            if isinstance(action, PassThrough):
                decision, decision_raw = True, raw
            elif isinstance(action, Consume):
                decision = False
            elif isinstance(action, TypeLetters):
                self._send_letters(action)
                injected = True
            elif isinstance(action, Retract):
                self.emitter.backspace(action.count)
                injected = True
            elif isinstance(action, CommitChord):
                self._commit(action)
                injected = True

        if decision is None:
            return False

        if decision and event is not None:
            # The composer gets first refusal on the keystroke. It may take it over
            # entirely, which is how the space after an automatic one is dropped and how
            # punctuation is pulled back against the preceding word. Asking before
            # answering the tap is the point: letting the key through as well would type
            # the character twice.
            consumed, wrote = self._observe(event)
            if consumed:
                return False
            injected = injected or wrote

        if decision and injected:
            # Keep the order. Our text has already gone out, so this keystroke follows it
            # through the same path rather than racing it.
            if decision_raw is not None:
                self.emitter.repost(decision_raw, synthetic_up=False)
            elif event is not None:
                self.emitter.type_text(_with_case(event))
            return False

        return decision

    def _send_letters(self, action: TypeLetters) -> None:
        """Send held-back letters, restoring the case the user actually typed.

        The letters are rebuilt from the recorded modifier state rather than by replaying
        the original events. For plain letters the two are indistinguishable to the
        application, and rebuilding keeps this path free of any dependence on macOS event
        objects, so it can be tested.
        """
        self.stats.letters += len(action.events)
        self.composer.invalidate()
        text = "".join(_with_case(held) for held in action.events)
        if text:
            self.emitter.type_text(text)
        if self.on_text is not None:
            self.on_text(text)

    def _commit(self, action: CommitChord) -> None:
        candidates = self.predict(action)
        if not candidates:
            self.stats.no_candidates += 1
            self.composer.invalidate()
            # Nothing contains these letters, so give the user what they pressed rather
            # than swallowing the keystrokes entirely.
            self.emitter.type_text(action.press_order)
            if self.on_text is not None:
                self.on_text(action.press_order)
            return

        self.stats.chords += 1
        word = candidates[0]
        for operation in self.composer.commit(word, candidates[:CANDIDATES]):
            self._apply(operation)
        if self.on_text is not None:
            self.on_text(word + " ")

    def _apply(self, operation) -> None:
        if isinstance(operation, TypeText):
            self.emitter.type_text(operation.text)
        elif isinstance(operation, Backspaces):
            self.emitter.backspace(operation.count)

    # -- repairing the last word ----------------------------------------------

    def _try_repair(self, event: KeyEvent) -> str | None:
        """Intercept the cycle and undo keys, but only while a commit is still live.

        Returns "handled" when the keystroke was consumed, "pass" when the repair keys are
        not applicable and it should go to the application, or None when this is not a
        repair key at all.
        """
        live = self.composer.last is not None and self.composer.last.live
        if not live or not event.is_down:
            return None

        if self.cycle_keycode is not None and event.keycode == self.cycle_keycode:
            operations = self.composer.cycle(forward=not event.shift)
            if not operations:
                return "pass"
            self.stats.cycles += 1
            for operation in operations:
                self._apply(operation)
            return "handled"

        if self.undo_keycode is not None and event.keycode == self.undo_keycode:
            operations = self.composer.undo()
            if not operations:
                return "pass"
            self.stats.undos += 1
            for operation in operations:
                self._apply(operation)
            return "handled"

        return None

    def _observe(self, event: KeyEvent) -> tuple[bool, bool]:
        """Follow what the user typed, and notice when the caret may have moved.

        Returns (was the keystroke consumed here, did anything get injected).
        """
        if not event.is_down:
            return (False, False)
        if event.keycode in NAVIGATION or event.command:
            self.composer.invalidate()
            if self.on_nav is not None:
                self.on_nav("navigate")
            return (False, False)

        character = self._character(event)
        if character is None:
            return (False, False)
        consumed, operations = self.composer.on_character(character)
        for operation in operations:
            self._apply(operation)
        if self.on_text is not None and not consumed:
            self.on_text(character)
        return (consumed, bool(operations))

    def _character(self, event: KeyEvent) -> str | None:
        if event.key:
            return event.key
        # Space sits at the same key code on every layout, and it is the one non-letter
        # the composer must never miss, since it decides whether an automatic space gets
        # doubled.
        if event.keycode == SPACE:
            return " "
        if self.layout is not None:
            return self.layout.to_char.get(event.keycode)
        return None
