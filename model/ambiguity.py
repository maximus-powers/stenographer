"""How ambiguous is a chord?

Accuracy depends first on how many words a chord could possibly mean. This module
measures that directly over the real vocabulary, weighted by how often each word actually
occurs, so the numbers describe the chords a user would really type rather than an average
over the dictionary. It also reports how many chords ask one finger to press two keys,
which is the difference between a brief that is comfortable and one that is not.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from model.candidates import CandidateIndex
from model.eval import Trial


@dataclass
class AmbiguityStats:
    size: int
    trials: int
    median: float
    p90: float
    maximum: int
    target_is_most_frequent: float
    same_finger_share: float


def measure(trials: Sequence[Trial], index: CandidateIndex) -> dict[int, AmbiguityStats]:
    """Candidate-set statistics grouped by chord size."""
    buckets: dict[int, list[int]] = {}
    wins: Counter[int] = Counter()
    same_finger: Counter[int] = Counter()

    for trial in trials:
        candidates = index.candidates(trial.request.letters)
        buckets.setdefault(trial.size, []).append(int(candidates.size))
        if candidates.size and index.vocab.words[candidates[0]].lower() == trial.target:
            wins[trial.size] += 1
        if trial.same_finger:
            same_finger[trial.size] += 1

    stats: dict[int, AmbiguityStats] = {}
    for size, counts in sorted(buckets.items()):
        ordered = sorted(counts)
        stats[size] = AmbiguityStats(
            size=size,
            trials=len(ordered),
            median=statistics.median(ordered),
            p90=ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
            maximum=ordered[-1],
            target_is_most_frequent=wins[size] / len(ordered),
            same_finger_share=same_finger[size] / len(ordered),
        )
    return stats


def render(stats: dict[int, AmbiguityStats]) -> str:
    lines = ["== chord ambiguity over the real vocabulary =="]
    lines.append(
        "  size      n   median   p90     max   target most frequent   two keys one finger"
    )
    for size, row in sorted(stats.items()):
        lines.append(
            f"  {size:>4d} {row.trials:>7,} {row.median:>8.0f} {row.p90:>6.0f} "
            f"{row.maximum:>7,} {row.target_is_most_frequent:>17.1%} {row.same_finger_share:>21.1%}"
        )
    return "\n".join(lines)


def worst_chords(trials: Sequence[Trial], index: CandidateIndex, limit: int = 10) -> str:
    """The most frequently typed chords whose candidates are hardest to separate."""
    seen: Counter[str] = Counter()
    for trial in trials:
        seen[trial.request.letters] += 1

    rows = []
    for letters, count in seen.most_common(60):
        candidates = index.candidates(letters)
        rows.append((count, letters, [index.vocab.words[i] for i in candidates[:6]]))
    rows.sort(key=lambda row: -row[0])

    lines = ["== most common chords and what they could mean =="]
    for count, letters, words in rows[:limit]:
        lines.append(f"  {{{','.join(sorted(letters))}}} x{count:<5d} {', '.join(words)}")
    return "\n".join(lines)
