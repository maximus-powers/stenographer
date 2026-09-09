"""Corpus reading: public text to (context, target word) examples.

Both the offline harness and the training pipeline need the same view of a corpus, a
stream of word tokens with their surrounding context, so that lives here. Only public
corpora are used; nothing the user types is ever read or stored.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from model.vocab import normalise

# A token is a word, possibly with internal apostrophes, or a single punctuation mark.
# Hyphens split so both halves become chordable words in their own right.
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[^\sA-Za-z]")

SENTENCE_END = frozenset(".!?")


@dataclass(frozen=True)
class Example:
    """One prediction site: the word to be produced and the text around it."""

    target: str  # surface form as it appeared in the corpus
    left: list[str]  # preceding tokens, nearest last
    right: list[str]  # following tokens, nearest first
    sentence_start: bool


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def iter_lines(source: str | Path, *, split: str = "test", limit: int | None = None) -> Iterator[str]:
    """Yield raw lines from a local text file or a Hugging Face wikitext split."""
    path = Path(source)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for count, line in enumerate(handle):
                if limit is not None and count >= limit:
                    return
                yield line
        return

    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", str(source), split=split)
    for count, row in enumerate(dataset):
        if limit is not None and count >= limit:
            return
        yield row["text"]


def iter_examples(
    lines: Iterable[str],
    *,
    left_window: int = 48,
    right_window: int = 16,
    min_left: int = 0,
) -> Iterator[Example]:
    """Walk the corpus and emit an Example at every word position.

    Context does not cross a blank line, since wikitext uses those as section breaks and
    text across one is not something a typist would have on screen as one field.
    """
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("="):
            continue
        tokens = tokenise(stripped)
        for position, token in enumerate(tokens):
            word = normalise(token)
            if word is None:
                continue
            left = tokens[max(0, position - left_window) : position]
            if len(left) < min_left:
                continue
            previous = next((t for t in reversed(left) if t.strip()), None)
            yield Example(
                target=token,
                left=left,
                right=tokens[position + 1 : position + 1 + right_window],
                sentence_start=previous is None or previous in SENTENCE_END,
            )


def count_words(lines: Iterable[str]) -> tuple[Counter[str], Counter[str]]:
    """Corpus frequencies and mid-sentence capitalisation counts, for vocabulary building.

    A word is counted as capitalised only away from a sentence start, which is what
    distinguishes a proper noun from an ordinary word that happened to open a sentence.
    """
    counts: Counter[str] = Counter()
    capitalised: Counter[str] = Counter()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("="):
            continue
        tokens = tokenise(stripped)
        at_start = True
        for token in tokens:
            word = normalise(token)
            if word is None:
                at_start = at_start and token not in SENTENCE_END
                if token in SENTENCE_END:
                    at_start = True
                continue
            lower = word.lower()
            counts[lower] += 1
            if not at_start and word[0].isupper():
                capitalised[lower] += 1
            at_start = False
    return counts, capitalised
