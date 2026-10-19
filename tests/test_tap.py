"""Tap-level tests for the hold-to-chord key.

These need Quartz, because the thing under test is how the tap reads macOS's modifier
flags. The detector's own behaviour is tested without it in test_detector.py.

What matters here is one distinction: right Option is the default hold-to-chord key, and
holding it must not look like a shortcut, or every chord made under it would be thrown
away. Left Option must still look like a shortcut. Both raise the same Alternate flag, so
the difference is only in the device bits.
"""

from __future__ import annotations

import pytest

Quartz = pytest.importorskip("Quartz")

from stenographer.keyboard.tap import (
    _LEFT_OPTION_DEVICE,
    _RIGHT_OPTION_DEVICE,
    EventTap,
)

ALTERNATE = Quartz.kCGEventFlagMaskAlternate
COMMAND = Quartz.kCGEventFlagMaskCommand
FN = Quartz.kCGEventFlagMaskSecondaryFn


def tap(chord_key: str | None) -> EventTap:
    return EventTap(lambda event, raw: True, listen_only=True, chord_key=chord_key)


def test_right_option_is_not_a_shortcut_when_it_is_the_chord_key():
    assert tap("right-option")._is_shortcut(ALTERNATE | _RIGHT_OPTION_DEVICE) is False


def test_left_option_is_still_a_shortcut():
    assert tap("right-option")._is_shortcut(ALTERNATE | _LEFT_OPTION_DEVICE) is True


def test_both_options_down_is_a_shortcut():
    flags = ALTERNATE | _LEFT_OPTION_DEVICE | _RIGHT_OPTION_DEVICE
    assert tap("right-option")._is_shortcut(flags) is True


def test_right_option_is_a_shortcut_when_it_is_not_the_chord_key():
    # No trigger selected it, so nothing should be exempt.
    assert tap(None)._is_shortcut(ALTERNATE | _RIGHT_OPTION_DEVICE) is True
    assert tap("fn")._is_shortcut(ALTERNATE | _RIGHT_OPTION_DEVICE) is True


def test_command_is_a_shortcut_even_while_the_chord_key_is_held():
    flags = COMMAND | ALTERNATE | _RIGHT_OPTION_DEVICE
    assert tap("right-option")._is_shortcut(flags) is True


def test_fn_is_never_a_shortcut():
    # Fn raises no shortcut flag, so it needs no exemption to work as a chord key.
    assert tap("fn")._is_shortcut(FN) is False


def test_plain_letters_are_not_a_shortcut():
    assert tap("right-option")._is_shortcut(0) is False


@pytest.mark.parametrize("chord_key,flags,expected", [
    ("right-option", ALTERNATE | _RIGHT_OPTION_DEVICE, True),
    ("right-option", ALTERNATE | _LEFT_OPTION_DEVICE, False),
    ("fn", FN, True),
    ("fn", ALTERNATE | _RIGHT_OPTION_DEVICE, False),
    (None, FN, False),
])
def test_chord_modifier_is_read_from_the_right_bits(chord_key, flags, expected):
    subject = tap(chord_key)
    assert bool(subject.chord_mask and flags & subject.chord_mask) is expected
