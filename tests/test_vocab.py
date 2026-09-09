import numpy as np
import pytest

from model.vocab import (
    Vocab,
    canonical_case,
    distinct_letters,
    letter_mask,
    normalise,
)


@pytest.mark.parametrize(
    "word,expected",
    [("the", set("the")), ("letter", set("letr")), ("a", {"a"}), ("", set())],
)
def test_letter_mask_encodes_distinct_letters(word, expected):
    mask = letter_mask(word)
    assert mask.bit_count() == len(expected)


def test_letter_mask_ignores_case_and_punctuation():
    assert letter_mask("Don't") == letter_mask("dont")


def test_distinct_letters_preserves_first_appearance_order():
    assert distinct_letters("letter") == "letr"
    assert distinct_letters("Mississippi") == "misp"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("the", "the"),
        ("Hello", "Hello"),
        ("don't", "don't"),
        ("a", None),  # single letters are typed directly, never chorded
        ("3rd", None),
        ("e-mail", None),  # the corpus reader splits hyphens instead
        ("café", None),
        ("", None),
        ("'tis", "tis"),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_canonical_case_needs_enough_evidence():
    # A word capitalised nearly always, and often enough to trust, is a proper noun.
    assert canonical_case("london", 0.95, 200) == "London"
    # The same share on a handful of sightings is noise, not evidence.
    assert canonical_case("great", 0.95, 3) == "great"
    # A word that is usually lowercase stays lowercase however often it is seen.
    assert canonical_case("may", 0.4, 500) == "may"


def test_canonical_case_handles_the_pronoun():
    assert canonical_case("i", 0.0, 0) == "I"


def test_vocab_round_trip(tmp_path):
    words = ["the", "and", "there"]
    vocab = Vocab(
        words=words,
        masks=np.asarray([letter_mask(w) for w in words], dtype=np.uint32),
        freqs=np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
    )
    vocab.save(tmp_path)
    loaded = Vocab.load(tmp_path)
    assert loaded.words == words
    assert np.array_equal(loaded.masks, vocab.masks)
    assert loaded.rank("THE") == 0
    assert loaded.rank("missing") is None
