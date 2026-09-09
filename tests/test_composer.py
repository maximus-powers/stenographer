"""Composer tests, checked against a simulated screen.

Every repair works by counting backspaces, and an off-by-one there eats a character of
the user's text or leaves a stray one behind. So rather than asserting on the operations,
these tests apply them to a text buffer and check what the user would actually see. That
catches arithmetic mistakes the operation list alone would hide.
"""

from __future__ import annotations

import pytest

from stenographer.keyboard.composer import (
    Backspaces,
    ComposerConfig,
    TextComposer,
    TypeText,
)


class Screen:
    """A text field, as far as the composer's operations can tell."""

    def __init__(self, text: str = "") -> None:
        self.text = text

    def apply(self, ops) -> Screen:
        for op in ops:
            if isinstance(op, TypeText):
                self.text += op.text
            elif isinstance(op, Backspaces):
                assert op.count <= len(self.text), (
                    f"tried to delete {op.count} characters from {self.text!r}"
                )
                if op.count:
                    self.text = self.text[: -op.count]
        return self

    def type(self, characters: str, composer: TextComposer) -> Screen:
        """Type printable characters, giving the composer first refusal on each."""
        for character in characters:
            handled, ops = composer.on_character(character)
            self.apply(ops)
            if not handled and not ops:
                self.text += character
        return self


@pytest.fixture
def composer():
    return TextComposer()


# -- committing -----------------------------------------------------------------


def test_a_committed_word_is_followed_by_a_space(composer):
    screen = Screen().apply(composer.commit("there"))
    assert screen.text == "there "


def test_the_automatic_space_is_on_by_default():
    assert ComposerConfig().auto_space is True


def test_consecutive_words_are_spaced_without_the_user_typing_anything():
    composer = TextComposer()
    screen = Screen()
    for word in ["I", "went", "there", "today"]:
        screen.apply(composer.commit(word))
    assert screen.text == "I went there today "


def test_a_leading_space_is_added_when_asked(composer):
    screen = Screen("hello").apply(composer.commit("there", needs_leading_space=True))
    assert screen.text == "hello there "


def test_auto_space_can_be_switched_off():
    composer = TextComposer(ComposerConfig(auto_space=False))
    screen = Screen().apply(composer.commit("there"))
    assert screen.text == "there"


# -- cycling through candidates -------------------------------------------------


def test_cycling_replaces_the_word_and_keeps_the_space(composer):
    screen = Screen().apply(composer.commit("there", ["there", "their", "they"]))
    screen.apply(composer.cycle())
    assert screen.text == "their "
    screen.apply(composer.cycle())
    assert screen.text == "they "


def test_cycling_wraps_around(composer):
    screen = Screen().apply(composer.commit("there", ["there", "their"]))
    screen.apply(composer.cycle())
    screen.apply(composer.cycle())
    assert screen.text == "there "


def test_cycling_backwards(composer):
    screen = Screen().apply(composer.commit("there", ["there", "their", "they"]))
    screen.apply(composer.cycle(forward=False))
    assert screen.text == "they "


def test_cycling_preserves_surrounding_text(composer):
    screen = Screen("I went ").apply(composer.commit("there", ["there", "their"]))
    screen.apply(composer.cycle())
    assert screen.text == "I went their "


def test_cycling_does_nothing_without_alternatives(composer):
    screen = Screen().apply(composer.commit("there", ["there"]))
    assert composer.cycle() == []
    assert screen.text == "there "


def test_cycling_does_nothing_once_the_emission_is_stale(composer):
    Screen().apply(composer.commit("there", ["there", "their"]))
    composer.invalidate()
    assert composer.cycle() == []


# -- undo -----------------------------------------------------------------------


def test_undo_removes_the_whole_word_and_its_space(composer):
    screen = Screen("I went ").apply(composer.commit("there"))
    screen.apply(composer.undo())
    assert screen.text == "I went "


def test_undo_after_punctuation_removes_everything_it_added(composer):
    screen = Screen("I went ").apply(composer.commit("there"))
    screen.type(",", composer)
    screen.apply(composer.undo())
    assert screen.text == "I went "


def test_undo_only_works_once(composer):
    screen = Screen("x ").apply(composer.commit("there"))
    screen.apply(composer.undo())
    assert composer.undo() == []
    assert screen.text == "x "


def test_falling_back_to_the_raw_letters(composer):
    screen = Screen().apply(composer.commit("there", ["there", "their"]))
    screen.apply(composer.replace_with_letters("eht"))
    assert screen.text == "eht"


# -- punctuation ----------------------------------------------------------------


@pytest.mark.parametrize("mark", [",", ".", ";", ":", "!", "?", ")", "]", "}"])
def test_punctuation_closes_up_against_the_word(composer, mark):
    screen = Screen().apply(composer.commit("there"))
    screen.type(mark, composer)
    assert screen.text == f"there{mark} "


def test_an_apostrophe_leaves_no_trailing_space(composer):
    screen = Screen().apply(composer.commit("dogs"))
    screen.type("'", composer)
    assert screen.text == "dogs'"


def test_a_word_typed_after_punctuation_is_spaced_correctly(composer):
    screen = Screen().apply(composer.commit("there"))
    screen.type(",", composer)
    screen.apply(composer.commit("then"))
    assert screen.text == "there, then "


def test_two_punctuation_marks_in_a_row(composer):
    screen = Screen().apply(composer.commit("really"))
    screen.type("?!", composer)
    assert screen.text == "really?! "


def test_punctuation_pull_back_can_be_switched_off():
    composer = TextComposer(ComposerConfig(pull_back_punctuation=False))
    screen = Screen().apply(composer.commit("there"))
    screen.type(",", composer)
    assert screen.text == "there ,"


def test_cycling_after_punctuation_keeps_the_punctuation(composer):
    screen = Screen().apply(composer.commit("there", ["there", "their"]))
    screen.type(",", composer)
    screen.apply(composer.cycle())
    assert screen.text == "their, "


# -- spaces ---------------------------------------------------------------------


def test_a_typed_space_after_the_automatic_one_is_swallowed(composer):
    screen = Screen().apply(composer.commit("there"))
    screen.type(" ", composer)
    assert screen.text == "there "


def test_only_the_first_extra_space_is_swallowed(composer):
    screen = Screen().apply(composer.commit("there"))
    screen.type("  ", composer)
    assert screen.text == "there  "


def test_space_collapsing_can_be_switched_off():
    composer = TextComposer(ComposerConfig(collapse_double_space=False))
    screen = Screen().apply(composer.commit("there"))
    screen.type(" ", composer)
    assert screen.text == "there  "


def test_an_ordinary_letter_ends_the_repair_window(composer):
    screen = Screen().apply(composer.commit("there"))
    screen.type("x", composer)
    assert screen.text == "there x"
    assert composer.cycle() == []


# -- arithmetic invariant -------------------------------------------------------


@pytest.mark.parametrize("word", ["a" * 2, "there", "understand", "extraordinarily"])
@pytest.mark.parametrize("marks", ["", ",", ".", "?!"])
def test_undo_always_restores_the_original_text(word, marks):
    """Whatever was typed, undo must leave the field exactly as it was.

    This is the invariant that matters most: a repair that deletes one character too
    many silently eats the user's own text.
    """
    original = "some earlier text "
    composer = TextComposer()
    screen = Screen(original).apply(composer.commit(word, [word, "other"]))
    screen.type(marks, composer)
    screen.apply(composer.undo())
    assert screen.text == original


@pytest.mark.parametrize("marks", ["", ",", "."])
def test_cycling_repeatedly_never_drifts(marks):
    candidates = ["there", "their", "they", "them"]
    original = "I went "
    composer = TextComposer()
    screen = Screen(original).apply(composer.commit("there", candidates))
    screen.type(marks, composer)
    for _ in range(len(candidates) * 3):
        screen.apply(composer.cycle())
    # A whole number of laps returns to the first candidate.
    assert screen.text == f"{original}there{marks} "
