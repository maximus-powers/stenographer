"""Vocabulary construction: frequency-ranked words with canonical casing and letter masks.

The vocabulary is the single source of truth shared by candidate generation, the
fallback index in the tap process, and the model's output head. Words are stored in
descending frequency order, which is what lets the fallback index pick the most
frequent candidate by taking the lowest set bit of a bitmask.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Letters that can form a chord. A chord is a set of these.
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
LETTER_INDEX = {c: i for i, c in enumerate(ALPHABET)}

# Words whose canonical form is capitalised regardless of position. Kept small and
# explicit; the model never sees case, so casing is applied by rule at emission time.
ALWAYS_CAPITALISED = {"i", "i'm", "i've", "i'll", "i'd"}

DEFAULT_SIZE = 50_000
MIN_LENGTH = 2


@dataclass(frozen=True)
class Vocab:
    """A frequency-ordered word list plus the bit masks used for candidate lookup."""

    words: list[str]  # canonical casing, descending frequency
    masks: np.ndarray  # uint32, one 26-bit letter set per word
    freqs: np.ndarray  # float32, Zipf-scale frequency, descending

    def __len__(self) -> int:
        return len(self.words)

    @property
    def index(self) -> dict[str, int]:
        if not hasattr(self, "_index"):
            object.__setattr__(self, "_index", {w.lower(): i for i, w in enumerate(self.words)})
        return self._index  # type: ignore[attr-defined]

    def rank(self, word: str) -> int | None:
        return self.index.get(word.lower())

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "vocab.json").write_text(
            json.dumps({"words": self.words, "freqs": [round(f, 4) for f in self.freqs.tolist()]})
        )
        np.save(directory / "letters.npy", self.masks)

    @classmethod
    def load(cls, directory: Path | str) -> Vocab:
        directory = Path(directory)
        payload = json.loads((directory / "vocab.json").read_text())
        masks = np.load(directory / "letters.npy")
        return cls(
            words=payload["words"],
            masks=masks,
            freqs=np.asarray(payload["freqs"], dtype=np.float32),
        )


def letter_mask(word: str) -> int:
    """26-bit set of the distinct a-z letters in `word`."""
    mask = 0
    for ch in word.lower():
        bit = LETTER_INDEX.get(ch)
        if bit is not None:
            mask |= 1 << bit
    return mask


def distinct_letters(word: str) -> str:
    """Distinct a-z letters of `word`, in first-appearance order."""
    seen: list[str] = []
    for ch in word.lower():
        if ch in LETTER_INDEX and ch not in seen:
            seen.append(ch)
    return "".join(seen)


def normalise(raw: str) -> str | None:
    """Fold a raw token to a vocabulary candidate, or None if it is not a word.

    Accepts letters plus internal apostrophes ("don't"). Hyphenated forms are rejected
    here; the corpus reader splits them so both halves can be chorded separately.
    """
    word = unicodedata.normalize("NFKC", raw).strip().strip("'’")
    if not word:
        return None
    word = word.replace("’", "'")
    if len(word) < MIN_LENGTH:
        # Single letters are typed directly, never chorded, so they are not vocabulary.
        return None
    body = word.replace("'", "")
    if not body.isalpha() or not body.isascii():
        return None
    if "''" in word:
        return None
    return word


# Evidence required before treating a word as a proper noun. A word has to be
# capitalised mid-sentence nearly every time it appears, and appear often enough for that
# to mean something: Wikipedia text capitalises "Great" and "May" inside titles often
# enough to fool a loose threshold, and wrongly capitalising a common word is far more
# visible to the user than leaving a rare proper noun lowercase.
CAPITAL_SHARE = 0.85
CAPITAL_MIN_COUNT = 10


def canonical_case(word: str, capitalised_share: float, count: int = 0) -> str:
    """Pick the casing to emit for `word`.

    `capitalised_share` is the fraction of mid-sentence occurrences that were capitalised
    and `count` the number of occurrences behind that fraction.
    """
    lower = word.lower()
    if lower in ALWAYS_CAPITALISED:
        return lower.replace("i", "I", 1)
    if count >= CAPITAL_MIN_COUNT and capitalised_share >= CAPITAL_SHARE:
        return lower[0].upper() + lower[1:]
    return lower


def build(
    size: int = DEFAULT_SIZE,
    *,
    counts: dict[str, int] | None = None,
    capitalised: dict[str, int] | None = None,
    restrict: bool = False,
) -> Vocab:
    """Build a vocabulary from wordfreq, ordered by frequency.

    `counts` maps lowercase word to corpus frequency and `capitalised` to the number of
    mid-sentence capitalised uses, which is what separates proper nouns from ordinary
    words. Set `restrict` to keep only words the corpus actually contains; that needs a
    corpus large enough to cover the target size, so it is off by default.
    """
    from wordfreq import top_n_list, zipf_frequency

    pool = top_n_list("en", max(size * 4, 200_000), wordlist="best")
    chosen: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw in pool:
        word = normalise(raw)
        if word is None:
            continue
        lower = word.lower()
        if lower in seen:
            continue
        if restrict and counts is not None and lower not in counts:
            continue
        if len(distinct_letters(lower)) < 2:
            # Cannot be chorded: no two distinct letters to press together.
            continue
        seen.add(lower)
        chosen.append((lower, zipf_frequency(lower, "en", wordlist="best")))
        if len(chosen) >= size:
            break

    chosen.sort(key=lambda pair: -pair[1])
    words: list[str] = []
    for lower, _ in chosen:
        share, total = 0.0, 0
        if capitalised is not None and counts is not None:
            total = counts.get(lower, 0)
            if total:
                share = capitalised.get(lower, 0) / total
        words.append(canonical_case(lower, share, total))

    masks = np.asarray([letter_mask(w) for w in words], dtype=np.uint32)
    freqs = np.asarray([f for _, f in chosen], dtype=np.float32)
    return Vocab(words=words, masks=masks, freqs=freqs)
