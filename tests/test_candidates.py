import numpy as np
import pytest

from model.candidates import CandidateIndex, FallbackIndex, chord_mask, opening_mask
from model.vocab import Vocab, letter_mask

# Frequency order, most frequent first, which is what vocabulary order means.
WORDS = ["the", "and", "there", "they", "their", "ethanol", "heat", "earth", "cat", "little"]


def make_vocab(words):
    return Vocab(
        words=words,
        masks=np.asarray([letter_mask(w) for w in words], dtype=np.uint32),
        freqs=np.asarray(range(len(words), 0, -1), dtype=np.float32),
    )


@pytest.fixture
def vocab():
    return make_vocab(WORDS)


def brute_force(words, letters):
    """Every word whose first k distinct letters are exactly the chord."""
    wanted = set(letters)
    found = []
    for word in words:
        opening = list(dict.fromkeys(word.lower()))[: len(wanted)]
        if len(opening) == len(wanted) and set(opening) == wanted:
            found.append(word)
    return found


@pytest.mark.parametrize("letters", ["eht", "th", "ehrt", "aht", "xyz", "ae", "act"])
def test_numpy_index_matches_brute_force(vocab, letters):
    index = CandidateIndex(vocab)
    found = [vocab.words[i] for i in index.candidates(letters)]
    assert found == brute_force(WORDS, letters)


@pytest.mark.parametrize("letters", ["eht", "th", "ehrt", "act"])
def test_fallback_index_matches_brute_force(vocab, letters):
    fallback = FallbackIndex.from_vocab(vocab)
    expected = brute_force(WORDS, letters)
    assert fallback.top_k(letters, k=len(WORDS)) == expected
    assert fallback.count(letters) == len(expected)


def test_a_word_that_merely_contains_the_letters_is_not_a_candidate(vocab):
    """The whole point of the change.

    "heat" contains e, h and t, and under the old superset matching it was a candidate
    for every one of them. It opens h-e-a, so it is not what {e,h,t} means any more.
    """
    index = CandidateIndex(vocab)
    found = [vocab.words[i] for i in index.candidates("eht")]
    assert "heat" not in found
    assert set("eht") <= set("heat")  # it does still contain them
    assert found == ["the", "there", "they", "their", "ethanol"]


def test_a_doubled_letter_does_not_consume_a_key(vocab):
    """The first four of "little" are l, i, t, e: a hand cannot press "t" twice at once."""
    assert opening_mask("little", 4) == chord_mask("lite")
    index = CandidateIndex(vocab)
    assert [vocab.words[i] for i in index.candidates("lite")] == ["little"]


def test_a_word_shorter_than_the_chord_cannot_match(vocab):
    """"the" has three distinct letters, so it is not reachable with four keys."""
    assert opening_mask("the", 4) is None
    index = CandidateIndex(vocab)
    assert "the" not in [vocab.words[i] for i in index.candidates("ehtr")]


def test_fallback_best_is_the_most_frequent_candidate(vocab):
    fallback = FallbackIndex.from_vocab(vocab)
    # Vocabulary order is descending frequency, so "the" outranks the other t-h-e words.
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
    # {e,h,t} opens five words, so the relaxed path must not widen the set.
    assert np.array_equal(index.relaxed_candidates("eht"), index.candidates("eht"))
    # {e,h,z} opens nothing, so asking what two of the three would have opened beats
    # returning nothing at all.
    assert index.candidates("ehz").size == 0
    assert index.relaxed_candidates("ehz").size > 0


def test_bucket_contents_stay_in_frequency_order():
    """Lookup does no sorting, so the buckets have to be built in the right order."""
    words = ["cat", "cart", "crate", "tracker"]
    index = CandidateIndex(make_vocab(words))
    # "tracker" opens t-r-a-c, the same four letters, so it belongs here too - last,
    # because it is the least frequent of the three.
    assert [words[i] for i in index.candidates("cart")] == ["cart", "crate", "tracker"]
    fallback = FallbackIndex(words)
    assert fallback.best("cart") == "cart"
