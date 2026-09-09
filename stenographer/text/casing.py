"""Choosing how a predicted word should be capitalised.

The model works on lowercase text and a chord carries no case at all, so casing is
decided by rule at the moment the word is typed. Three signals feed in, in order of
authority: the word's own canonical form from the vocabulary, what the user did with
Shift and Caps Lock, and where the caret sits in the sentence.

Getting this wrong is unusually visible. Capitalising an ordinary word mid-sentence looks
like a bug, so where the evidence is thin the rules stay lowercase.
"""

from __future__ import annotations

from dataclasses import dataclass

SENTENCE_ENDINGS = frozenset(".!?")
# An opening double quote introduces speech, which conventionally starts a sentence of
# its own: in `He said "` the next word is capitalised.
QUOTE_OPENERS = frozenset('"“«')
# Brackets and the apostrophe say nothing either way, so look straight past them. An
# apostrophe is here rather than with the quotes because it is far more often part of a
# word than the start of speech.
TRANSPARENT_OPENERS = frozenset("([{'‘")


@dataclass(frozen=True)
class CasingContext:
    """What is known about the caret's surroundings when a word is typed."""

    preceding: str = ""  # text just before the caret, as far back as is known
    shift: bool = False
    caps_lock: bool = False
    known: bool = True  # False when the preceding text could not be read at all


def starts_sentence(preceding: str) -> bool:
    """Would a word typed here begin a sentence?

    Only spaces and tabs are trimmed, never newlines: a line break is itself evidence of
    a fresh start, and stripping it would throw that away.
    """
    trimmed = preceding.rstrip(" \t")
    if not trimmed:
        return True
    if trimmed.endswith(("\n", "\r")):
        return True
    while trimmed and trimmed[-1] in TRANSPARENT_OPENERS:
        trimmed = trimmed[:-1].rstrip(" \t")
        if trimmed.endswith(("\n", "\r")):
            return True
    if not trimmed:
        return True
    if trimmed[-1] in QUOTE_OPENERS:
        return True
    return trimmed[-1] in SENTENCE_ENDINGS


def apply(word: str, context: CasingContext, *, code_context: bool = False) -> str:
    """Case `word` for this position.

    `word` arrives in its canonical vocabulary form, so a proper noun is already
    capitalised and keeps that regardless of position. `code_context` switches off
    sentence-start capitalisation, which is wrong in a terminal or an editor where a
    capitalised identifier is a different identifier.
    """
    if context.caps_lock:
        return word.upper()
    if context.shift:
        return word[:1].upper() + word[1:]
    if word[:1].isupper():
        return word  # canonical proper noun
    if code_context or not context.known:
        return word
    if starts_sentence(context.preceding):
        return word[:1].upper() + word[1:]
    return word
