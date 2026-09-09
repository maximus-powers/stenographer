"""Casing tests.

Capitalising a word that should be lowercase is one of the most visible ways this tool
can look broken, so the rules err towards lowercase whenever the evidence is thin. These
tests pin down where each signal wins.
"""

from __future__ import annotations

import pytest

from stenographer.text.casing import CasingContext, apply, starts_sentence


@pytest.mark.parametrize(
    "preceding,expected",
    [
        ("", True),
        ("   ", True),
        ("The cat sat.", True),
        ("Really?", True),
        ("Stop!", True),
        ("The cat", False),
        ("the cat ", False),
        ("a comma,", False),
        ("line one\n", True),
        ('He said "', True),  # a quote opens what is still a new sentence
        ("It ended. (", True),
        ("mid (", False),
    ],
)
def test_sentence_start_detection(preceding, expected):
    assert starts_sentence(preceding) is expected


def test_a_word_at_a_sentence_start_is_capitalised():
    assert apply("there", CasingContext(preceding="I stopped.")) == "There"


def test_a_word_mid_sentence_stays_lowercase():
    assert apply("there", CasingContext(preceding="I went")) == "there"


def test_a_canonical_proper_noun_keeps_its_capital_anywhere():
    assert apply("London", CasingContext(preceding="I went to")) == "London"


def test_shift_capitalises_an_ordinary_word():
    assert apply("there", CasingContext(preceding="I went", shift=True)) == "There"


def test_caps_lock_uppercases_the_whole_word():
    assert apply("there", CasingContext(preceding="I went", caps_lock=True)) == "THERE"


def test_caps_lock_beats_shift():
    context = CasingContext(preceding="", shift=True, caps_lock=True)
    assert apply("there", context) == "THERE"


def test_unknown_context_stays_lowercase():
    """With no idea where the caret is, the safer guess is lowercase."""
    assert apply("there", CasingContext(preceding="", known=False)) == "there"


def test_shift_still_works_without_context():
    context = CasingContext(preceding="", known=False, shift=True)
    assert apply("there", context) == "There"


def test_code_context_suppresses_sentence_capitalisation():
    # A full stop would capitalise this in prose; in code it is a member access.
    context = CasingContext(preceding="value = compute().")
    assert apply("total", context, code_context=True) == "total"
    assert apply("total", context) == "Total"


def test_code_context_still_honours_an_explicit_shift():
    context = CasingContext(preceding="x = ", shift=True)
    assert apply("total", context, code_context=True) == "Total"


def test_single_letter_words_are_handled():
    assert apply("I", CasingContext(preceding="and")) == "I"
