import numpy as np
import pytest

from model.candidates import CandidateIndex, FallbackIndex, chord_mask
from model.vocab import Vocab, letter_mask

WORDS = ["the", "and", "there", "they", "their", "ethanol", "heat", "earth", "cat"]


@pytest.fixture
def vocab():
    return Vocab(
        words=WORDS,
        masks=np.asarray([letter_mask(w) for w in WORDS], dtype=np.uint32),
        freqs=np.asarray(range(len(WORDS), 0, -1), dtype=np.float32),
    )


def brute_force(words, letters):
    return [w for w in words if set(letters) <= set(w.lower())]


@pytest.mark.parametrize("letters", ["eht", "aht", "tr", "ehrt", "xyz", "ae"])
def test_numpy_index_matches_brute_force(vocab, letters):
    index = CandidateIndex(vocab)
    found = [vocab.words[i] for i in index.candidates(letters)]
    assert found == brute_force(WORDS, letters)


@pytest.mark.parametrize("letters", ["eht", "aht", "tr", "ehrt", "ae"])
def test_fallback_index_matches_brute_force(vocab, letters):
    fallback = FallbackIndex.from_vocab(vocab)
    expected = brute_force(WORDS, letters)
    assert fallback.top_k(letters, k=len(WORDS)) == expected
    assert fallback.count(letters) == len(expected)


def test_fallback_best_is_the_most_frequent_candidate(vocab):
    fallback = FallbackIndex.from_vocab(vocab)
    # Vocabulary order is descending frequency, so "the" outranks the other {e,h,t} words.
    assert fallback.best("eht") == "the"
    assert fallback.best("xyz") is None


def test_chord_letters_are_order_independent(vocab):
    index = CandidateIndex(vocab)
    assert np.array_equal(index.candidates("eht"), index.candidates("the"))
    assert np.array_equal(index.candidates({"t", "h", "e"}), index.candidates("hte"))
    assert chord_mask("the") == chord_mask({"e", "h", "t"})


def test_repeated_chord_letters_collapse(vocab):
    index = CandidateIndex(vocab)
    assert np.array_equal(index.candidates("tthe"), index.candidates("the"))


def test_relaxed_matching_only_fires_when_strict_finds_nothing(vocab):
    index = CandidateIndex(vocab)
    # {e,h,t} matches plenty, so the relaxed path must not widen the set.
    assert np.array_equal(index.relaxed_candidates("eht"), index.candidates("eht"))
    # {e,h,z} matches nothing, so dropping a letter is better than failing.
    assert index.candidates("ehz").size == 0
    assert index.relaxed_candidates("ehz").size > 0
