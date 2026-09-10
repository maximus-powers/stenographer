"""Synthetic chord generation.

The model never sees the user's own typing, so its training signal comes entirely from
chords synthesised over a public corpus. What that signal has to look like is decided by
candidate generation, not by guesswork about habits: a chord is a word's first k distinct
letters, so `prefix` is the only policy that produces a chord the deployed system can
resolve, and it carries all the weight.

The other three policies are kept because the evaluation harness reports accuracy per
policy, and a number for the briefs this no longer accepts is worth having when the
question of widening the window comes back. They are not sampled during training. An
earlier mixture spent 65% of its examples on them, which taught the model to rank words
against a candidate set that no longer exists.

Substitution noise, which models a finger landing one key off, is available but off by
default, and the reason is worth recording. Candidate generation keeps a word only when
the chord is exactly its opening letters, so a chord carrying a letter the target does
not open with excludes that target outright. The model's output is masked to the candidate set, so it could
never name the intended word however well it was trained; training on such examples only
teaches it to back wrong words. Relaxed matching does not rescue them either: it fires
only when nothing matched at all, and answers with what a smaller chord would have
opened, which is not the intended word. A slipped finger is therefore handled where it belongs, in the cycle and
undo keys, rather than pretended away in the training distribution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from model.vocab import distinct_letters

VOWELS = frozenset("aeiou")

# Which finger types each key on a US QWERTY layout. Two chord letters sharing a finger
# force an awkward same-column reach, so those chords are down-weighted in training and
# reported separately by the ambiguity harness.
FINGERS: dict[str, str] = {}
for _finger, _keys in {
    "L4": "qaz",
    "L3": "wsx",
    "L2": "edc",
    "L1": "rfvtgb",
    "R1": "yhnujm",
    "R2": "ik",
    "R3": "ol",
    "R4": "p",
}.items():
    for _key in _keys:
        FINGERS[_key] = _finger

# Physically adjacent keys, used to model a finger landing one key off target.
NEIGHBOURS: dict[str, str] = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn",
    "j": "hukm", "k": "jil", "l": "kop", "m": "nj", "n": "bhm", "b": "vgn",
    "v": "cfb", "c": "xdv", "x": "zsc", "z": "ax",
}

SIZE_WEIGHTS = {2: 0.10, 3: 0.40, 4: 0.35, 5: 0.15}
POLICY_WEIGHTS = {"prefix": 1.0}


@dataclass
class SynthConfig:
    """Knobs for chord synthesis, all overridable from the training config."""

    size_weights: dict[int, float] = field(default_factory=lambda: dict(SIZE_WEIGHTS))
    policy_weights: dict[str, float] = field(default_factory=lambda: dict(POLICY_WEIGHTS))
    # Off by default: see the module docstring. A chord with an off-target letter is
    # unreachable through candidate generation, so training on one is counter-productive.
    substitution_noise: float = 0.0
    # Also off by default. Rerolling an awkward chord swaps in a uniformly random subset,
    # which discards the very prefix or skeleton structure that made the chord
    # predictable, and measured accuracy on rerolled chords is less than half that of the
    # policies they replace. Typability is reported by the ambiguity harness instead, and
    # what the user's hands actually prefer gets measured by the M1 chord drill.
    typability_resample: float = 0.0
    min_word_letters: int = 2


def same_finger_pairs(letters: str | set[str]) -> int:
    """How many pairs of chord letters share a finger."""
    used: dict[str, int] = {}
    for ch in set(letters):
        finger = FINGERS.get(ch)
        if finger:
            used[finger] = used.get(finger, 0) + 1
    return sum(n * (n - 1) // 2 for n in used.values())


def is_typable(letters: str | set[str]) -> bool:
    return same_finger_pairs(letters) == 0


def _weighted_choice(rng: random.Random, weights: dict) -> object:
    total = sum(weights.values())
    threshold = rng.random() * total
    running = 0.0
    for key, weight in weights.items():
        running += weight
        if threshold <= running:
            return key
    return next(reversed(weights))


def policy_prefix(letters: str, size: int, rng: random.Random) -> str:
    """The word's first `size` distinct letters, the most intuitive brief."""
    return letters[:size]


def policy_skeleton(letters: str, size: int, rng: random.Random) -> str:
    """First letter plus consonants in order, the way shorthand usually drops vowels."""
    head, rest = letters[0], letters[1:]
    consonants = [c for c in rest if c not in VOWELS]
    picked = [head] + consonants[: size - 1]
    if len(picked) < size:
        picked += [c for c in rest if c not in picked][: size - len(picked)]
    return "".join(picked[:size])


def policy_outer(letters: str, size: int, rng: random.Random) -> str:
    """First and last letter plus filler from the middle."""
    if len(letters) <= size:
        return letters[:size]
    picked = [letters[0], letters[-1]]
    middle = letters[1:-1]
    if middle and size > 2:
        picked += rng.sample(list(middle), min(size - 2, len(middle)))
    return "".join(dict.fromkeys(picked))[:size]


def policy_random(letters: str, size: int, rng: random.Random) -> str:
    return "".join(rng.sample(list(letters), min(size, len(letters))))


POLICIES = {
    "prefix": policy_prefix,
    "skeleton": policy_skeleton,
    "outer": policy_outer,
    "random": policy_random,
}


def sample_chord(
    word: str,
    rng: random.Random,
    config: SynthConfig | None = None,
    *,
    policy: str | None = None,
    size: int | None = None,
) -> tuple[str, str] | None:
    """Draw one chord for `word`.

    Returns (chord letters sorted, effective policy), or None when the word has too few
    distinct letters to chord. `policy` and `size` force a choice, which the evaluation
    harness uses to report accuracy per policy and per chord size. The returned policy is
    the one that actually produced the chord: a chord rerolled for typability comes back
    as "resampled" and one perturbed by noise as "noisy", so accuracy tables never credit
    a policy for a chord it did not generate.
    """
    config = config or SynthConfig()
    letters = distinct_letters(word)
    if len(letters) < max(2, config.min_word_letters):
        return None

    chosen_policy = policy or _weighted_choice(rng, config.policy_weights)
    if size is None:
        weights = {k: v for k, v in config.size_weights.items() if k <= len(letters)}
        if not weights:
            weights = {len(letters): 1.0}
        size = _weighted_choice(rng, weights)
    size = min(size, len(letters))

    chord = POLICIES[str(chosen_policy)](letters, int(size), rng)

    effective = str(chosen_policy)
    if (
        config.typability_resample
        and not is_typable(chord)
        and rng.random() < config.typability_resample
    ):
        alternative = policy_random(letters, int(size), rng)
        if is_typable(alternative):
            chord, effective = alternative, "resampled"

    if config.substitution_noise and rng.random() < config.substitution_noise:
        chord = _substitute(chord, rng, word)
        effective = "noisy"

    return "".join(sorted(set(chord))), effective


def _substitute(chord: str, rng: random.Random, word: str = "") -> str:
    """Replace one chord letter with a physical neighbour, modelling a slipped finger.

    When `word` is given, only neighbours absent from the word are used, so the result is
    genuinely off target rather than accidentally still valid.
    """
    chars = list(chord)
    position = rng.randrange(len(chars))
    banned = set(chars) | set(word.lower())
    options = [c for c in NEIGHBOURS.get(chars[position], "") if c not in banned]
    if not options:
        options = [c for c in NEIGHBOURS.get(chars[position], "") if c not in chars]
    if options:
        chars[position] = rng.choice(options)
    return "".join(chars)
