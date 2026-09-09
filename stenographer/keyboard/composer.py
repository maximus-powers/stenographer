"""Turning a committed word into keystrokes, and repairing it afterwards.

Typing the word is the easy part. What makes chorded input usable is everything around
it: the space that follows automatically, the punctuation that has to slip in front of
that space, and the two keys that fix a wrong guess without breaking the user's flow.

All of it hangs on knowing exactly what was last emitted, so that a repair removes
precisely the right number of characters. That bookkeeping lives here, deliberately
separated from the event tap so it can be tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Punctuation that belongs tight against the preceding word, so the automatic space has
# to be taken back before it is typed.
TIGHT_PUNCTUATION = frozenset(",.;:!?)]}\"'")
# Punctuation that attaches without wanting a space afterwards either.
NO_SPACE_AFTER = frozenset("'-/")


class Op:
    """One instruction for the emitter."""


@dataclass(frozen=True)
class TypeText(Op):
    text: str


@dataclass(frozen=True)
class Backspaces(Op):
    count: int


@dataclass
class Emission:
    """The last word this composer typed, and how to take it back."""

    word: str
    candidates: tuple[str, ...]
    index: int = 0
    suffix: str = ""  # punctuation typed after the word
    auto_space: bool = True
    live: bool = True

    @property
    def width(self) -> int:
        """Characters on screen, which is how many backspaces undo it."""
        return len(self.word) + len(self.suffix) + (1 if self.auto_space else 0)


@dataclass(frozen=True)
class ComposerConfig:
    auto_space: bool = True
    collapse_double_space: bool = True
    pull_back_punctuation: bool = True


@dataclass
class TextComposer:
    """Emits committed words and handles the keys that follow them."""

    config: ComposerConfig = field(default_factory=ComposerConfig)
    last: Emission | None = None

    # -- committing -----------------------------------------------------------

    def commit(
        self,
        word: str,
        candidates: tuple[str, ...] | list[str] = (),
        *,
        needs_leading_space: bool = False,
    ) -> list[Op]:
        """Type `word`, with a space after it and, if asked, one before.

        `candidates` are the alternatives the cycle key steps through, best first.
        """
        ordered = tuple(candidates) or (word,)
        prefix = " " if needs_leading_space else ""
        trailing = " " if self.config.auto_space else ""
        self.last = Emission(
            word=word,
            candidates=ordered,
            index=max(0, list(ordered).index(word)) if word in ordered else 0,
            auto_space=self.config.auto_space,
        )
        return [TypeText(prefix + word + trailing)]

    # -- repair ---------------------------------------------------------------

    def cycle(self, forward: bool = True) -> list[Op]:
        """Replace the last word with the next candidate."""
        emission = self.last
        if emission is None or not emission.live or len(emission.candidates) < 2:
            return []
        step = 1 if forward else -1
        emission.index = (emission.index + step) % len(emission.candidates)
        replacement = emission.candidates[emission.index]
        removed = emission.width
        emission.word = replacement
        trailing = " " if emission.auto_space else ""
        return [Backspaces(removed), TypeText(replacement + emission.suffix + trailing)]

    def undo(self) -> list[Op]:
        """Remove the last word entirely, leaving nothing in its place."""
        emission = self.last
        if emission is None or not emission.live:
            return []
        self.last = None
        return [Backspaces(emission.width)]

    def replace_with_letters(self, letters: str) -> list[Op]:
        """Swap the last word for the raw chord letters, when no candidate was right."""
        emission = self.last
        if emission is None or not emission.live:
            return []
        removed = emission.width
        self.last = None
        return [Backspaces(removed), TypeText(letters)]

    # -- what the user types next --------------------------------------------

    def on_character(self, character: str) -> tuple[bool, list[Op]]:
        """Handle a printable character typed straight after a commit.

        Returns whether the keystroke was handled here, and the operations to perform.
        Handled means the key must not also reach the application: a space swallowed
        after an automatic one is handled with no operations at all, which is different
        from a key this composer has no interest in.
        """
        emission = self.last
        if emission is None or not emission.live:
            return False, []

        if character == " ":
            emission.live = False
            # The automatic space already covers this one, so drop the keystroke.
            return (self.config.collapse_double_space and emission.auto_space), []

        if character in TIGHT_PUNCTUATION and self.config.pull_back_punctuation:
            if not emission.auto_space:
                emission.suffix += character
                return False, []
            emission.suffix += character
            if character in NO_SPACE_AFTER:
                emission.auto_space = False
                return True, [Backspaces(1), TypeText(character)]
            # Keep the word joined to its punctuation, and put the space back after it so
            # the next word does not need one of its own.
            return True, [Backspaces(1), TypeText(character + " ")]

        emission.live = False
        return False, []

    def invalidate(self) -> None:
        """Forget the last emission, because the caret may have moved."""
        if self.last is not None:
            self.last.live = False
