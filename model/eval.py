"""Offline evaluation harness.

This is milestone zero's whole point: measure whether a chord can be resolved accurately
enough to be worth typing, before a single line of keyboard code exists. Every predictor
is scored the same way on held-out public text, broken down by the factors that actually
move accuracy: how many letters the chord has, which briefing habit produced it, and how
common the target word is.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from model.corpus import Example
from model.synth import SynthConfig, same_finger_pairs, sample_chord
from model.vocab import Vocab, distinct_letters
from stenographer.brain.predictor.base import Predictor, Request

# Vocabulary-rank bands. Accuracy varies enormously across these, and the top band is
# both the most frequent and the most ambiguous, so a single average hides the story.
BANDS: tuple[tuple[str, int, int], ...] = (
    ("top 100", 0, 100),
    ("100-1k", 100, 1_000),
    ("1k-10k", 1_000, 10_000),
    ("10k+", 10_000, 1 << 30),
)


def band_for(rank: int | None) -> str:
    if rank is None:
        return "out of vocabulary"
    for name, low, high in BANDS:
        if low <= rank < high:
            return name
    return "10k+"


@dataclass(frozen=True)
class Trial:
    """One scored prediction site."""

    target: str  # lowercase target word
    request: Request
    policy: str
    size: int
    band: str
    length: int
    same_finger: int


@dataclass
class Bucket:
    trials: int = 0
    top1: int = 0
    top3: int = 0
    top5: int = 0

    def add(self, rank: int | None) -> None:
        self.trials += 1
        if rank is None:
            return
        if rank < 1:
            self.top1 += 1
        if rank < 3:
            self.top3 += 1
        if rank < 5:
            self.top5 += 1

    def rates(self) -> tuple[float, float, float]:
        if not self.trials:
            return (0.0, 0.0, 0.0)
        n = self.trials
        return (self.top1 / n, self.top3 / n, self.top5 / n)


@dataclass
class Report:
    """Accuracy tables for one predictor over one trial set."""

    name: str
    overall: Bucket = field(default_factory=Bucket)
    by_size: dict[int, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    by_policy: dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    by_band: dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    by_context: dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    ranks: Counter[int] = field(default_factory=Counter)

    def add(self, trial: Trial, rank: int | None) -> None:
        self.overall.add(rank)
        self.by_size[trial.size].add(rank)
        self.by_policy[trial.policy].add(rank)
        self.by_band[trial.band].add(rank)
        context = "no context" if not trial.request.left else (
            "short (1-7)" if len(trial.request.left) < 8 else "long (8+)"
        )
        self.by_context[context].add(rank)
        self.ranks[-1 if rank is None else rank] += 1

    def render(self) -> str:
        lines = [f"== {self.name} =={'':2s}({self.overall.trials:,} trials)"]
        top1, top3, top5 = self.overall.rates()
        lines.append(f"  overall      top1 {top1:6.1%}   top3 {top3:6.1%}   top5 {top5:6.1%}")

        def table(title: str, buckets: dict, keys: Sequence) -> None:
            lines.append(f"  {title}")
            for key in keys:
                bucket = buckets.get(key)
                if not bucket or not bucket.trials:
                    continue
                a, b, c = bucket.rates()
                label = str(key)
                lines.append(
                    f"    {label:<14s} n={bucket.trials:>7,}  "
                    f"top1 {a:6.1%}  top3 {b:6.1%}  top5 {c:6.1%}"
                )

        table("by chord size", self.by_size, sorted(self.by_size))
        table("by context length", self.by_context, ["no context", "short (1-7)", "long (8+)"])
        table("by word frequency", self.by_band, [name for name, _, _ in BANDS])
        table("by chord policy", self.by_policy, sorted(self.by_policy))
        return "\n".join(lines)


def build_trials(
    examples: Iterable[Example],
    vocab: Vocab,
    *,
    limit: int = 20_000,
    seed: int = 0,
    config: SynthConfig | None = None,
    policy: str | None = None,
    size: int | None = None,
    min_word_length: int = 2,
) -> tuple[list[Trial], Counter[str]]:
    """Turn corpus examples into scored trials, and count why examples were skipped.

    Only targets present in the vocabulary become trials, because a chord can never
    resolve to a word the vocabulary does not hold. The skip counter is what turns that
    into an honest coverage number rather than a silently inflated accuracy.
    """
    rng = random.Random(seed)
    config = config or SynthConfig()
    trials: list[Trial] = []
    skipped: Counter[str] = Counter()

    for example in examples:
        lower = example.target.lower()
        skipped["tokens seen"] += 1
        if len(lower) < min_word_length:
            skipped["too short to chord"] += 1
            continue
        if len(distinct_letters(lower)) < 2:
            skipped["fewer than 2 distinct letters"] += 1
            continue
        rank = vocab.rank(lower)
        if rank is None:
            skipped["out of vocabulary"] += 1
            continue
        drawn = sample_chord(lower, rng, config, policy=policy, size=size)
        if drawn is None:
            skipped["no chord possible"] += 1
            continue
        letters, effective = drawn
        skipped["chorded"] += 1
        trials.append(
            Trial(
                target=lower,
                request=Request(
                    letters=letters,
                    left=tuple(example.left),
                    right=tuple(example.right),
                    sentence_start=example.sentence_start,
                ),
                policy=effective,
                size=len(letters),
                band=band_for(rank),
                length=len(lower),
                same_finger=same_finger_pairs(letters),
            )
        )
        if len(trials) >= limit:
            break

    return trials, skipped


def score(predictor: Predictor, trials: Sequence[Trial], name: str, k: int = 5) -> Report:
    report = Report(name=name)
    for trial in trials:
        ranked = [w.lower() for w in predictor.predict(trial.request, k)]
        rank = ranked.index(trial.target) if trial.target in ranked else None
        report.add(trial, rank)
    return report
