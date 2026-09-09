"""End-to-end tests of the keystroke path, without macOS.

The detector, composer and emitter are each tested on their own, but the bugs that would
actually reach a user live in how they fit together: a letter typed twice, a word that
eats the character after it, a repair key that stops working. So these tests drive the
real `Runtime` with a fake emitter and assert on the text a field would end up holding.

The emitter records injected text into a buffer, and passed-through keystrokes are
appended to the same buffer, which is what the application would see.
"""

from __future__ import annotations

import pytest

from stenographer.keyboard.composer import TextComposer
from stenographer.keyboard.detector import ChordDetector, DetectorConfig
from stenographer.keyboard.events import KeyEvent, Kind, down, up
from stenographer.keyboard.layout import BACKSPACE, RETURN, TAB
from stenographer.runtime import Runtime

VOCAB = {
    "eht": ["the", "they", "their", "there", "them"],
    "ot": ["to", "not", "out", "other"],
    "bces": ["because", "describes", "subject"],
    "mrtw": ["tomorrow", "warmth"],
    "gnst": ["something", "against", "strong"],
    "adn": ["and", "hand", "around"],
}


class FakeEmitter:
    """Stands in for the macOS emitter, recording what would reach the application."""

    def __init__(self) -> None:
        self.text = ""
        self.operations: list[str] = []

    def type_text(self, text: str) -> None:
        if not text:
            return
        self.text += text
        self.operations.append(f"type {text!r}")

    def backspace(self, count: int) -> None:
        if count <= 0:
            return
        assert count <= len(self.text), f"deleting {count} from {self.text!r}"
        self.text = self.text[:-count]
        self.operations.append(f"backspace {count}")

    def repost(self, raw, synthetic_up: bool = True) -> None:
        # The raw object is the character the test passed in as a stand-in for a CGEvent.
        self.text += str(raw)
        self.operations.append(f"repost {raw!r}")


class FakeLayout:
    """The few non-letter keys these tests need to produce characters."""

    to_char = {49: " ", 43: ",", 47: ".", 41: ";", 39: "'"}


class Harness:
    """Drives a Runtime the way the event tap does, firing timers as deadlines pass."""

    def __init__(self, **config) -> None:
        self.emitter = FakeEmitter()
        self.composer = TextComposer()
        self.detector = ChordDetector(DetectorConfig(**config))
        self.runtime = Runtime(
            detector=self.detector,
            composer=self.composer,
            emitter=self.emitter,
            predict=lambda chord: VOCAB.get(chord.letters, []),
            layout=FakeLayout(),
        )

    def feed(self, *events: KeyEvent) -> Harness:
        for event in events:
            deadline = self.runtime.next_deadline()
            while deadline is not None and deadline <= event.t_ms:
                self.runtime.on_timer(deadline)
                following = self.runtime.next_deadline()
                if following == deadline:
                    break
                deadline = following
            keep = self.runtime.handle(event, raw=self._raw(event))
            if keep and event.is_down:
                self.emitter.text += self._raw(event)
        return self

    def settle(self) -> Harness:
        deadline = self.runtime.next_deadline()
        if deadline is not None:
            self.runtime.on_timer(deadline + 1)
        return self

    @staticmethod
    def _raw(event: KeyEvent) -> str:
        """What the application would receive for a passed-through keystroke."""
        if event.key:
            return event.key.upper() if event.shift != event.caps_lock else event.key
        return {TAB: "\t", RETURN: "\n", BACKSPACE: ""}.get(event.keycode, " ")

    @property
    def text(self) -> str:
        return self.emitter.text


def chord(*keys: str, start: float = 0.0, spread: float = 10.0, hold: float = 150.0, **kwargs):
    """A deliberate simultaneous press of `keys`."""
    events = [down(key, start + index * spread, **kwargs) for index, key in enumerate(keys)]
    events += [up(key, start + hold + index * spread, **kwargs) for index, key in enumerate(keys)]
    return events


def typed(word: str, start: float = 0.0, interval: float = 120.0):
    """Ordinary letter-by-letter typing."""
    events = []
    for index, letter in enumerate(word):
        moment = start + index * interval
        events.append(down(letter, moment))
        events.append(up(letter, moment + 50))
    return events


def space(t_ms: float) -> KeyEvent:
    return KeyEvent(kind=Kind.DOWN, t_ms=t_ms, key=None, keycode=49)


# -- the basic path -------------------------------------------------------------


def test_a_chord_becomes_a_word_with_a_space():
    harness = Harness().feed(*chord("t", "h", "e")).settle()
    assert harness.text == "the "


def test_ordinary_typing_is_unchanged():
    harness = Harness().feed(*typed("hello")).settle()
    assert harness.text == "hello"


def test_typing_and_chording_mix():
    events = typed("I say ", interval=120)
    events += chord("t", "h", "e", start=800)
    harness = Harness().feed(*events).settle()
    assert harness.text == "I say the "


def test_two_chords_in_a_row():
    events = chord("t", "h", "e") + chord("b", "c", "e", "s", start=400)
    harness = Harness().feed(*events).settle()
    assert harness.text == "the because "


def test_a_chord_with_no_candidates_types_the_letters():
    harness = Harness().feed(*chord("x", "q", "j")).settle()
    assert harness.text == "xqj"


def test_case_is_preserved_when_a_group_is_typed_as_letters():
    harness = Harness().feed(down("h", 0, shift=True), up("h", 40)).settle()
    assert harness.text == "H"


def test_caps_lock_and_shift_cancel_out():
    harness = Harness().feed(down("h", 0, shift=True, caps_lock=True), up("h", 40)).settle()
    assert harness.text == "h"


# -- ordering -------------------------------------------------------------------


def test_letters_arrive_before_the_key_that_flushed_them():
    """A space that closes a group must land after the group's letters, not before."""
    harness = Harness().feed(down("t", 0), down("h", 12), space(20)).settle()
    assert harness.text == "th "


def test_a_word_then_a_typed_letter_keeps_its_order():
    events = chord("t", "h", "e") + typed("x", start=400)
    harness = Harness().feed(*events).settle()
    assert harness.text == "the x"


# -- punctuation and spacing ----------------------------------------------------

def test_punctuation_closes_up_against_a_committed_word():
    events = chord("t", "h", "e")
    harness = Harness().feed(*events).settle()
    harness.feed(KeyEvent(kind=Kind.DOWN, t_ms=400, key=None, keycode=43))
    assert harness.text == "the, "


def test_a_typed_space_after_a_committed_word_is_swallowed():
    harness = Harness().feed(*chord("t", "h", "e")).settle()
    harness.feed(space(400))
    assert harness.text == "the "


# -- repair ---------------------------------------------------------------------


def test_the_cycle_key_steps_through_candidates():
    harness = Harness().feed(*chord("t", "h", "e")).settle()
    harness.feed(down("", 400, keycode=TAB))
    assert harness.text == "they "
    harness.feed(down("", 500, keycode=TAB))
    assert harness.text == "their "


def test_the_cycle_key_is_swallowed_so_it_never_reaches_the_application():
    harness = Harness().feed(*chord("t", "h", "e")).settle()
    keep = harness.runtime.handle(down("", 400, keycode=TAB))
    assert keep is False
    assert "\t" not in harness.text


def test_shift_cycles_backwards():
    harness = Harness().feed(*chord("t", "h", "e")).settle()
    harness.feed(down("", 400, keycode=TAB, shift=True))
    assert harness.text == "them "


def test_backspace_removes_the_whole_committed_word():
    events = typed("I say ") + chord("t", "h", "e", start=800)
    harness = Harness().feed(*events).settle()
    harness.feed(down("", 1200, keycode=BACKSPACE))
    assert harness.text == "I say "


def test_tab_reaches_the_application_when_no_word_is_pending():
    harness = Harness().feed(*typed("x")).settle()
    keep = harness.runtime.handle(down("", 400, keycode=TAB))
    assert keep is True


def test_backspace_reaches_the_application_when_no_word_is_pending():
    harness = Harness().feed(*typed("x")).settle()
    keep = harness.runtime.handle(down("", 400, keycode=BACKSPACE))
    assert keep is True


def test_repair_stops_working_once_the_user_moves_on():
    harness = Harness().feed(*chord("t", "h", "e")).settle()
    harness.feed(*typed("x", start=400))
    keep = harness.runtime.handle(down("", 800, keycode=TAB))
    assert keep is True
    assert harness.text == "the x"


def test_arrow_keys_invalidate_the_pending_word():
    harness = Harness().feed(*chord("t", "h", "e")).settle()
    harness.runtime.handle(down("", 400, keycode=123))  # left arrow
    keep = harness.runtime.handle(down("", 500, keycode=TAB))
    assert keep is True


# -- nothing is lost -----------------------------------------------------------


@pytest.mark.parametrize("interval", [70, 90, 120, 200])
def test_every_typed_letter_reaches_the_application(interval):
    word = "keyboard"
    harness = Harness().feed(*typed(word, interval=interval)).settle()
    assert harness.text == word


def test_a_long_mixed_passage_loses_nothing():
    events = typed("we go ", interval=110)
    events += chord("t", "h", "e", start=700)
    events += typed("x ", start=1100, interval=110)
    events += chord("m", "r", "t", "w", start=1500)
    harness = Harness().feed(*events).settle()
    assert harness.text == "we go the x tomorrow "


def test_statistics_are_recorded():
    events = typed("ab", interval=150) + chord("t", "h", "e", start=500)
    harness = Harness().feed(*events).settle()
    assert harness.runtime.stats.chords == 1
    assert harness.runtime.stats.letters == 2
