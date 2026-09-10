"""Candidate generation: a chord is the opening letters of a word.

A chord of k keys is the word's **first k distinct letters**, in any order. Press `t`,
`h` and `e` and the candidates are the words that begin t-h-e in some order: the, they,
their, there, them. Not "other", which contains all three letters somewhere but does not
open with them.

That is a change of kind, not of degree. Matching every word that merely contains the
letters means a 4-key chord has around 600 candidates in a 50,000-word vocabulary and a
frequency ranker names the right one a quarter of the time. Requiring the letters to be
the opening ones leaves about 32, and the same ranker is right 45% of the time, measured
over 20,000 held-out prediction sites. The model has to separate far fewer words, and the
user has to remember nothing: the chord for a word is the start of the word.

It also makes matching exact rather than a filter. Since the chord has k letters and a
word's first k distinct letters are k letters, "contains" and "equals" are the same
question, so the chord's own bitmask is a dictionary key and the lookup is a hash rather
than a scan over the whole vocabulary. Both indexes below are that dictionary:

* `CandidateIndex` (numpy) lives in the brain and produces the boolean mask the model's
  output head is restricted to.
* `FallbackIndex` (pure Python) lives in the tap process, which must never import numpy
  or torch. It answers "most frequent candidate" with one dictionary hit, so a chord
  still resolves when the brain is dead or too slow.

Words are stored in vocabulary order, which is descending frequency, so the first entry
of a bucket is always its most frequent word and no sorting happens at lookup time.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from model.vocab import LETTER_INDEX, Vocab, letter_mask

# The largest chord any bucket is built for. The detector caps a chord at six keys; two
# spare sizes cost a few hundred kilobytes and mean raising --max-keys does not silently
# stop matching.
MAX_CHORD = 8


def chord_mask(letters: str | set[str]) -> int:
    """26-bit mask for a chord given as a string or a set of single characters."""
    if not isinstance(letters, str):
        letters = "".join(sorted(letters))
    return letter_mask(letters)


def opening_mask(word: str, size: int) -> int | None:
    """Mask of a word's first `size` distinct letters, or None if it has fewer.

    Distinct letters, not characters: the first four of "little" are l, i, t, e. Doubling
    a letter is something a word does, not something a hand can press twice at once, so
    counting characters here would make "little" unchordable at four keys.
    """
    seen: list[str] = []
    for char in word.lower():
        if char in LETTER_INDEX and char not in seen:
            seen.append(char)
            if len(seen) == size:
                return letter_mask("".join(seen))
    return None


def build_buckets(words: list[str], max_size: int = MAX_CHORD) -> dict[tuple[int, int], list[int]]:
    """Map (chord size, chord mask) to the vocabulary indices it names, best first."""
    buckets: dict[tuple[int, int], list[int]] = {}
    for position, word in enumerate(words):
        for size in range(2, max_size + 1):
            key = opening_mask(word, size)
            if key is None:
                # Fewer than `size` distinct letters, so no larger size can match either.
                break
            buckets.setdefault((size, key), []).append(position)
    return buckets


def _key(letters: str | set[str]) -> tuple[int, int]:
    chars = set(letters)
    return len(chars), chord_mask(chars)


class CandidateIndex:
    """Opening-letter buckets over a vocabulary."""

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab
        self.buckets = {
            key: np.asarray(rows, dtype=np.int64)
            for key, rows in build_buckets(vocab.words).items()
        }
        self._empty = np.empty(0, dtype=np.int64)

    def candidates(self, letters: str | set[str]) -> np.ndarray:
        """Vocabulary indices of every candidate, in descending frequency order."""
        return self.buckets.get(_key(letters), self._empty)

    def mask_for(self, letters: str | set[str]) -> np.ndarray:
        """Boolean array over the vocabulary: True where the word opens with the chord."""
        mask = np.zeros(len(self.vocab), dtype=bool)
        mask[self.candidates(letters)] = True
        return mask

    def relaxed_candidates(self, letters: str | set[str], drop: int = 1) -> np.ndarray:
        """Candidates allowing up to `drop` chord letters to be wrong.

        This matters more than it used to. Under "contains" matching a chord with one
        wrong letter still matched plenty of words, so this path almost never ran; under
        opening-letter matching a wrong letter usually matches nothing at all, and the
        honest recovery is to ask what the remaining letters would have opened.
        """
        strict = self.candidates(letters)
        chars = sorted(set(letters))
        if strict.size or drop <= 0 or len(chars) <= 2:
            return strict
        found = [self.candidates("".join(subset)) for subset in combinations(chars, len(chars) - drop)]
        joined = [rows for rows in found if rows.size]
        if not joined:
            return self._empty
        return np.unique(np.concatenate(joined))

    def top_k(self, letters: str | set[str], k: int = 5) -> list[str]:
        """The k most frequent candidates, which is the untrained baseline predictor."""
        rows = self.candidates(letters)
        if rows.size == 0:
            rows = self.relaxed_candidates(letters)
        return [self.vocab.words[i] for i in rows[:k]]


class FallbackIndex:
    """The same buckets without numpy, for the process holding the event tap."""

    def __init__(self, words: list[str]) -> None:
        self.words = words
        self.buckets = build_buckets(words)

    @classmethod
    def from_vocab(cls, vocab: Vocab) -> FallbackIndex:
        return cls(vocab.words)

    def candidates(self, letters: str | set[str]) -> list[int]:
        return self.buckets.get(_key(letters), [])

    def best(self, letters: str | set[str]) -> str | None:
        """Most frequent word opening with the chord letters, or None."""
        rows = self.candidates(letters)
        return self.words[rows[0]] if rows else None

    def top_k(self, letters: str | set[str], k: int = 5) -> list[str]:
        return [self.words[i] for i in self.candidates(letters)[:k]]

    def count(self, letters: str | set[str]) -> int:
        return len(self.candidates(letters))
