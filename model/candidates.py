"""Candidate generation: chord letter-set to the words that could have produced it.

A chord is a set of letters the user pressed together, and a word is a candidate when it
contains every one of them (steno-style briefs, not a prefix). Membership is a superset
test on 26-bit masks, so the whole vocabulary is filtered with one vectorised operation.

Two indexes exist because they serve different processes:

* `CandidateIndex` (numpy) lives in the brain and produces the boolean mask the model's
  output head is restricted to.
* `FallbackIndex` (pure Python big integers) lives in the tap process, which must never
  import numpy or torch. It answers "most frequent candidate" in one AND plus a
  lowest-set-bit, so a chord still resolves when the brain is dead or too slow.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from model.vocab import ALPHABET, LETTER_INDEX, Vocab, letter_mask


def chord_mask(letters: str | set[str]) -> int:
    """26-bit mask for a chord given as a string or a set of single characters."""
    if not isinstance(letters, str):
        letters = "".join(sorted(letters))
    return letter_mask(letters)


class CandidateIndex:
    """Superset filtering over a vocabulary's letter masks."""

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab
        self.masks = vocab.masks.astype(np.uint32, copy=False)

    def mask_for(self, letters: str | set[str]) -> np.ndarray:
        """Boolean array over the vocabulary: True where the word contains every letter."""
        needed = np.uint32(chord_mask(letters))
        return (self.masks & needed) == needed

    def candidates(self, letters: str | set[str]) -> np.ndarray:
        """Vocabulary indices of every candidate, in descending frequency order."""
        return np.flatnonzero(self.mask_for(letters))

    def relaxed_candidates(self, letters: str | set[str], drop: int = 1) -> np.ndarray:
        """Candidates allowing up to `drop` chord letters to be wrong.

        This is a last resort for a chord that matches nothing at all, so it fires only
        when strict matching comes back empty. It is not a cure for a mistyped letter: a
        chord with one wrong letter usually still matches plenty of other words, so the
        strict set is non-empty and this path never runs. Repairing a slipped finger is
        the job of the cycle and undo keys.
        """
        chars = sorted(set(letters))
        strict = self.candidates(chars)
        if strict.size or drop <= 0 or len(chars) <= 2:
            return strict
        keep = len(chars) - drop
        combined = np.zeros(len(self.vocab), dtype=bool)
        for subset in combinations(chars, keep):
            combined |= self.mask_for("".join(subset))
        return np.flatnonzero(combined)

    def top_k(self, letters: str | set[str], k: int = 5) -> list[str]:
        """The k most frequent candidates, which is the untrained baseline predictor."""
        idx = self.candidates(letters)
        if idx.size == 0:
            idx = self.relaxed_candidates(letters)
        return [self.vocab.words[i] for i in idx[:k]]


class FallbackIndex:
    """One big integer per letter, bit i set when vocabulary word i contains that letter.

    Bit order is vocabulary order, so the lowest set bit of an intersection is always the
    most frequent candidate.
    """

    def __init__(self, words: list[str]) -> None:
        self.words = words
        bits = {c: 0 for c in ALPHABET}
        for position, word in enumerate(words):
            for ch in set(word.lower()):
                if ch in LETTER_INDEX:
                    bits[ch] |= 1 << position
        self.bits = bits
        self.universe = (1 << len(words)) - 1

    @classmethod
    def from_vocab(cls, vocab: Vocab) -> FallbackIndex:
        return cls(vocab.words)

    def intersect(self, letters: str | set[str]) -> int:
        acc = self.universe
        for ch in set(letters):
            acc &= self.bits.get(ch, 0)
            if not acc:
                return 0
        return acc

    def best(self, letters: str | set[str]) -> str | None:
        """Most frequent word containing every chord letter, or None."""
        acc = self.intersect(letters)
        if not acc:
            return None
        return self.words[(acc & -acc).bit_length() - 1]

    def top_k(self, letters: str | set[str], k: int = 5) -> list[str]:
        acc = self.intersect(letters)
        out: list[str] = []
        while acc and len(out) < k:
            low = acc & -acc
            out.append(self.words[low.bit_length() - 1])
            acc ^= low
        return out

    def count(self, letters: str | set[str]) -> int:
        return self.intersect(letters).bit_count()
