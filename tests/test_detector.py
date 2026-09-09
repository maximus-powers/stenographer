"""Chord detector tests.

The detector takes no macOS dependency, so every case here is written as a timed sequence
of key events. That is deliberate: the difference between a chord and fast typing is
entirely a matter of milliseconds, and these are the numbers the whole idea rests on.

`run` drives the machine the way the real tap does, firing the timer whenever the
detector's own deadline has passed, so the tests exercise the same path as production.
"""

from __future__ import annotations

import pytest

from stenographer.keyboard.detector import (
    ChordDetector,
    CommitChord,
    Consume,
    DetectorConfig,
    PassThrough,
    Retract,
    Trigger,
    TypeLetters,
)
from stenographer.keyboard.events import KeyEvent, Kind, down, other, up


def run(detector: ChordDetector, events: list[KeyEvent], settle: bool = True) -> list:
    """Feed events in order, firing timers as their deadlines pass."""
    actions: list = []
    for event in events:
        deadline = detector.next_deadline()
        while deadline is not None and deadline <= event.t_ms:
            fired = detector.on_timer(deadline)
            actions.extend(fired)
            next_deadline = detector.next_deadline()
            if next_deadline == deadline:
                break
            deadline = next_deadline
        actions.extend(detector.feed(event))
    if settle:
        deadline = detector.next_deadline()
        if deadline is not None:
            actions.extend(detector.on_timer(deadline))
    return actions


def chords(actions: list) -> list[str]:
    return [a.letters for a in actions if isinstance(a, CommitChord)]


def typed(actions: list) -> str:
    return "".join(a.letters for a in actions if isinstance(a, TypeLetters))


def passed(actions: list) -> str:
    return "".join(
        a.event.key or "" for a in actions if isinstance(a, PassThrough) and a.event.is_down
    )


# -- chords ---------------------------------------------------------------------


def test_three_keys_pressed_together_is_a_chord():
    detector = ChordDetector()
    actions = run(
        detector,
        [down("t", 0), down("h", 12), down("e", 22), up("t", 140), up("h", 150), up("e", 160)],
    )
    assert chords(actions) == ["eht"]
    assert typed(actions) == ""


def test_chord_letters_are_sorted_and_press_order_is_kept_separately():
    detector = ChordDetector()
    actions = run(
        detector,
        [down("t", 0), down("h", 10), down("e", 20), up("t", 140), up("h", 150), up("e", 160)],
    )
    commit = next(a for a in actions if isinstance(a, CommitChord))
    assert commit.letters == "eht"
    assert commit.press_order == "the"


def test_a_late_finger_joins_while_nothing_has_been_released():
    detector = ChordDetector()
    actions = run(
        detector,
        [
            down("t", 0),
            down("h", 20),
            down("e", 40),
            down("r", 70),
            up("t", 200),
            up("h", 210),
            up("e", 220),
            up("r", 230),
        ],
    )
    assert chords(actions) == ["ehrt"]


def test_a_late_finger_is_rejected_once_a_key_has_been_released():
    detector = ChordDetector()
    actions = run(
        detector,
        [down("t", 0), down("h", 20), up("t", 55), down("r", 70), up("h", 200), up("r", 210)],
    )
    # The first pair overlapped only 35 ms, so it is rollover, and "r" starts fresh.
    assert chords(actions) == []
    assert typed(actions) == "thr"


def test_five_key_chord():
    detector = ChordDetector()
    events = [down(k, i * 8) for i, k in enumerate("stgnr")]
    events += [up(k, 200 + i * 8) for i, k in enumerate("stgnr")]
    assert chords(run(detector, events)) == ["gnrst"]


def test_more_keys_than_the_hand_can_hold_are_typed_as_letters():
    detector = ChordDetector(DetectorConfig(max_keys=4))
    events = [down(k, i * 8) for i, k in enumerate("abcde")]
    events.append(up("a", 300))
    actions = run(detector, events)
    assert chords(actions) == []
    assert "abcd" in typed(actions)


# -- single letters and ordinary typing -----------------------------------------


def test_a_single_tap_is_typed_normally():
    detector = ChordDetector()
    actions = run(detector, [down("a", 0), up("a", 30)])
    assert chords(actions) == []
    assert typed(actions) == "a"


def test_a_single_held_key_is_eventually_emitted_on_its_own():
    detector = ChordDetector()
    actions = run(detector, [down("a", 0)])
    letters = [a for a in actions if isinstance(a, TypeLetters)]
    assert letters and letters[0].letters == "a"
    # Still held, so the real release must reach the application and repeat keeps working.
    assert letters[0].synthetic_up is False


def test_a_released_key_is_emitted_at_once_not_after_the_hold_limit():
    """Normal typing must not wait for the chord budget to expire.

    A keystroke is decided when it is released, so the letter appears as the finger
    lifts. Getting this wrong would put a second of latency on every character.
    """
    detector = ChordDetector(DetectorConfig(lone_hold_ms=1000.0))
    actions = []
    for event in [down("a", 0), up("a", 70)]:
        actions.extend(detector.feed(event))
    assert typed(actions) == "a"
    assert detector.group is None


@pytest.mark.parametrize("interval", [60, 80, 100, 140, 200])
def test_sequential_typing_never_forms_a_chord(interval):
    detector = ChordDetector()
    events: list[KeyEvent] = []
    for position, key in enumerate("chord"):
        start = position * interval
        events.append(down(key, start))
        events.append(up(key, start + 40))
    actions = run(detector, events)
    assert chords(actions) == []
    assert typed(actions) == "chord"


@pytest.mark.parametrize("overlap", [10, 20, 30, 40])
def test_rollover_typing_is_not_a_chord(overlap):
    """Fast typists release the previous key shortly after the next goes down."""
    detector = ChordDetector()
    events: list[KeyEvent] = []
    interval = 70
    for position, key in enumerate("chord"):
        start = position * interval
        events.append(down(key, start))
        events.append(up(key, start + interval + overlap))
    actions = run(detector, events)
    assert chords(actions) == []
    assert typed(actions) == "chord"


def test_two_keys_held_together_long_enough_is_a_chord():
    detector = ChordDetector()
    actions = run(detector, [down("t", 0), down("o", 20), up("t", 150), up("o", 160)])
    assert chords(actions) == ["ot"]


def test_two_keys_with_too_little_overlap_are_letters():
    detector = ChordDetector()
    # "o" goes down at 20 and "t" is released at 60: only 40 ms of overlap.
    actions = run(detector, [down("t", 0), down("o", 20), up("t", 60), up("o", 200)])
    assert chords(actions) == []
    assert typed(actions) == "to"


def test_the_two_key_overlap_threshold_is_the_boundary():
    config = DetectorConfig(two_key_overlap_ms=60.0)
    just_short = ChordDetector(config)
    assert chords(run(just_short, [down("t", 0), down("o", 20), up("t", 79), up("o", 90)])) == []
    just_enough = ChordDetector(config)
    assert chords(
        run(just_enough, [down("t", 0), down("o", 20), up("t", 81), up("o", 90)])
    ) == ["ot"]


def test_a_second_press_after_the_window_starts_a_new_group():
    detector = ChordDetector()
    actions = run(detector, [down("a", 0), down("b", 200), up("a", 210), up("b", 240)])
    assert chords(actions) == []
    assert typed(actions) == "ab"


# -- a chord may be formed slowly -----------------------------------------------


@pytest.mark.parametrize("spread", [20, 150, 400, 700, 900])
def test_a_chord_can_be_formed_slowly(spread):
    """Fingers may arrive over most of a second, so long as none has left yet.

    This is what the one-second budget is for: there is no need to hit the keys at the
    same instant, only to have them all down together at some point before lifting.
    """
    detector = ChordDetector()
    step = spread / 2
    events = [down("t", 0), down("h", step), down("e", spread)]
    events += [up(k, spread + 40 + i * 10) for i, k in enumerate("the")]
    assert chords(run(detector, events)) == ["eht"]


def test_a_gesture_slower_than_the_limit_is_not_a_chord():
    detector = ChordDetector(DetectorConfig(chord_max_ms=1000.0))
    events = [down("t", 0), down("h", 400), down("e", 1400)]
    events += [up(k, 1500 + i * 10) for i, k in enumerate("the")]
    actions = run(detector, events)
    # "t" and "h" overlapped long enough to be a pair; "e" arrived too late to join them.
    assert "eht" not in chords(actions)


def test_keys_still_down_when_the_time_runs_out_are_typed_as_letters():
    detector = ChordDetector(DetectorConfig(chord_max_ms=1000.0))
    actions = run(detector, [down("t", 0), down("h", 20), down("e", 40)], settle=False)
    actions.extend(detector.on_timer(1200))
    assert chords(actions) == []
    assert typed(actions) == "the"


def test_a_roll_where_a_finger_leaves_early_is_not_a_chord():
    """Three keys touched in sequence, never all down together, is typing.

    This is the case a wide time limit would otherwise swallow: at 120 words per minute
    the keys of a word overlap in pairs, and only the requirement that every key be down
    at one instant tells that apart from a deliberate press.
    """
    detector = ChordDetector()
    actions = run(
        detector,
        [
            down("t", 0),
            down("h", 60),
            up("t", 90),  # gone before "e" arrives
            down("e", 120),
            up("h", 150),
            up("e", 200),
        ],
    )
    assert chords(actions) == []
    assert typed(actions) == "the"


def test_a_whole_word_typed_with_heavy_rollover_produces_no_chord():
    """Every consecutive pair overlaps, which is what a fast typist really does."""
    detector = ChordDetector()
    events: list[KeyEvent] = []
    interval = 60
    for position, key in enumerate("keyboard"):
        start = position * interval
        events.append(down(key, start))
        events.append(up(key, start + interval + 25))  # released 25 ms into the next key
    actions = run(detector, events)
    assert chords(actions) == []
    assert typed(actions) == "keyboard"


def test_the_time_limit_is_configurable():
    quick = ChordDetector(DetectorConfig(chord_max_ms=200.0))
    slow_gesture = [down("t", 0), down("h", 150), down("e", 300)]
    slow_gesture += [up(k, 400 + i * 10) for i, k in enumerate("the")]
    assert chords(run(quick, slow_gesture)) == []

    patient = ChordDetector(DetectorConfig(chord_max_ms=2000.0))
    assert chords(run(patient, slow_gesture)) == ["eht"]


# -- modifiers, non-letters, repeats --------------------------------------------


def test_shift_held_during_a_chord_is_reported():
    detector = ChordDetector()
    actions = run(
        detector,
        [
            down("t", 0, shift=True),
            down("h", 12, shift=True),
            down("e", 22),
            up("t", 150),
            up("h", 160),
            up("e", 170),
        ],
    )
    commit = next(a for a in actions if isinstance(a, CommitChord))
    assert commit.shift is True
    assert commit.caps_lock is False


def test_caps_lock_is_reported():
    detector = ChordDetector()
    actions = run(
        detector,
        [
            down("t", 0, caps_lock=True),
            down("h", 12, caps_lock=True),
            up("t", 150),
            up("h", 160),
        ],
    )
    commit = next(a for a in actions if isinstance(a, CommitChord))
    assert commit.caps_lock is True


def test_command_shortcuts_pass_straight_through():
    detector = ChordDetector()
    actions = run(detector, [down("a", 0, command=True), up("a", 40, command=True)])
    assert chords(actions) == []
    assert typed(actions) == ""
    assert passed(actions) == "a"


def test_a_non_letter_closes_an_open_group_before_passing():
    detector = ChordDetector()
    actions = run(detector, [down("t", 0), down("h", 12), other(20), up("t", 150)])
    # The space arrives before the group could be judged a chord, so the letters go out.
    assert typed(actions) == "th"
    assert any(isinstance(a, PassThrough) and a.event.key is None for a in actions)


def test_ordering_is_preserved_when_a_group_is_closed_by_a_non_letter():
    detector = ChordDetector()
    actions = run(detector, [down("t", 0), down("h", 12), other(20)])
    kinds = [type(a).__name__ for a in actions if not isinstance(a, Consume)]
    assert kinds == ["TypeLetters", "PassThrough"]


def test_autorepeat_inside_a_group_is_dropped():
    detector = ChordDetector()
    actions = run(
        detector,
        [
            down("t", 0),
            down("h", 12),
            down("t", 240, autorepeat=True),
            up("t", 300),
            up("h", 310),
        ],
    )
    assert chords(actions) == ["ht"]


def test_autorepeat_outside_a_group_passes_through():
    detector = ChordDetector()
    run(detector, [down("a", 0), up("a", 20)])
    actions = detector.feed(down("a", 240, autorepeat=True))
    assert [type(a).__name__ for a in actions] == ["PassThrough"]


def test_flags_changed_events_pass_through():
    detector = ChordDetector()
    actions = detector.feed(KeyEvent(kind=Kind.FLAGS, t_ms=0.0, shift=True))
    assert [type(a).__name__ for a in actions] == ["PassThrough"]


# -- resting fingers, lost events, resets ---------------------------------------


def test_keys_still_held_after_the_maximum_are_not_a_chord():
    """Fingers resting on the keyboard must not produce a word."""
    detector = ChordDetector(DetectorConfig(chord_max_ms=1000.0))
    actions = run(detector, [down("a", 0), down("s", 20), down("d", 40)], settle=False)
    actions.extend(detector.on_timer(1100))
    assert chords(actions) == []
    assert typed(actions) == "asd"


def test_a_lost_release_is_recovered_on_the_next_press_of_that_key():
    detector = ChordDetector()
    # "t" goes down twice with no release in between, which means a release was lost.
    actions = run(detector, [down("t", 0), down("h", 12), down("t", 200), up("t", 400)])
    assert typed(actions).count("t") >= 1
    assert detector.group is None or "t" in detector.group.held


def test_releases_of_held_back_keys_never_reach_the_application():
    detector = ChordDetector()
    for event in [down("t", 0), down("h", 12), down("e", 22)]:
        detector.feed(event)
    # None of the releases may pass through: the application never saw the presses.
    for event in [up("t", 150), up("h", 160), up("e", 170)]:
        actions = detector.feed(event)
        assert not any(isinstance(a, PassThrough) for a in actions), actions


def test_reset_flushes_an_open_group():
    detector = ChordDetector()
    detector.feed(down("t", 0))
    detector.feed(down("h", 12))
    actions = detector.reset()
    assert typed(actions) == "th"
    assert detector.group is None
    assert detector.pending_release == set()


def test_disabling_the_detector_passes_everything_through():
    detector = ChordDetector()
    detector.enabled = False
    actions = run(detector, [down("t", 0), down("h", 12), down("e", 22)])
    assert chords(actions) == []
    assert passed(actions) == "the"


# -- alternative triggers -------------------------------------------------------


def test_optimistic_mode_types_immediately_then_retracts_for_a_chord():
    detector = ChordDetector(DetectorConfig(trigger=Trigger.OPTIMISTIC))
    actions = run(
        detector,
        [down("t", 0), down("h", 12), down("e", 22), up("t", 150), up("h", 160), up("e", 170)],
    )
    assert passed(actions) == "the"
    retractions = [a for a in actions if isinstance(a, Retract)]
    assert [r.count for r in retractions] == [3]
    assert chords(actions) == ["eht"]


def test_optimistic_mode_leaves_ordinary_typing_alone():
    detector = ChordDetector(DetectorConfig(trigger=Trigger.OPTIMISTIC))
    events: list[KeyEvent] = []
    for position, key in enumerate("chord"):
        events.append(down(key, position * 90))
        events.append(up(key, position * 90 + 40))
    actions = run(detector, events)
    assert passed(actions) == "chord"
    assert not [a for a in actions if isinstance(a, Retract)]


def test_hold_to_chord_ignores_letters_until_the_modifier_is_down():
    detector = ChordDetector(DetectorConfig(trigger=Trigger.HOLD_TO_CHORD))
    actions = run(detector, [down("t", 0), down("h", 12), down("e", 22)])
    assert chords(actions) == []
    assert passed(actions) == "the"


def test_hold_to_chord_forms_a_chord_while_the_modifier_is_down():
    detector = ChordDetector(DetectorConfig(trigger=Trigger.HOLD_TO_CHORD))
    detector.set_chord_modifier(True, 0)
    actions = run(
        detector,
        [down("t", 10), down("h", 22), down("e", 32), up("t", 150), up("h", 160), up("e", 170)],
    )
    assert chords(actions) == ["eht"]


def test_releasing_the_hold_to_chord_modifier_commits_the_group():
    detector = ChordDetector(DetectorConfig(trigger=Trigger.HOLD_TO_CHORD))
    detector.set_chord_modifier(True, 0)
    for event in [down("t", 10), down("o", 20)]:
        detector.feed(event)
    actions = detector.set_chord_modifier(False, 100)
    assert chords(actions) == ["ot"]


# -- invariants -----------------------------------------------------------------


@pytest.mark.parametrize("interval", [15, 35, 55, 75, 95, 130])
@pytest.mark.parametrize("hold", [30, 60, 120])
def test_every_held_back_press_is_accounted_for_exactly_once(interval, hold):
    """No keystroke may be silently dropped, whatever the timing.

    Each press must end up either typed as a letter or folded into a chord. Anything else
    means characters vanish as the user types, which is the one failure mode that would
    make the tool unusable.
    """
    detector = ChordDetector()
    word = "stenog"
    events: list[KeyEvent] = []
    for position, key in enumerate(word):
        start = position * interval
        events.append(down(key, start))
        events.append(up(key, start + hold))
    actions = run(detector, events)

    letters_out = typed(actions) + passed(actions)
    chord_letters = "".join(chords(actions))
    accounted = len(letters_out) + len(chord_letters)
    assert accounted == len(word), (
        f"{len(word)} presses produced {len(letters_out)} letters and "
        f"{len(chord_letters)} chord keys"
    )


@pytest.mark.parametrize("interval", [60, 70, 80, 90, 110, 150])
def test_no_chord_when_presses_never_overlap(interval):
    detector = ChordDetector()
    events: list[KeyEvent] = []
    for position, key in enumerate("keyboard"):
        start = position * interval
        events.append(down(key, start))
        events.append(up(key, start + interval - 5))  # released before the next press
    assert chords(run(detector, events)) == []


def test_no_pending_state_is_left_behind_after_a_burst():
    detector = ChordDetector()
    events: list[KeyEvent] = []
    for position, key in enumerate("thequickbrownfox"):
        start = position * 75
        events.append(down(key, start))
        events.append(up(key, start + 50))
    run(detector, events)
    assert detector.group is None
    assert detector.pending_release == set()
