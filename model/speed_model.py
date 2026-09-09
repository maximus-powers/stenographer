"""Is chording actually faster than typing?

Keystroke savings alone do not answer that. A wrong first guess costs the user time to
notice and time to repair, and published work on word suggestion has found systems that
cut keystrokes while making able-bodied typists slower overall. This module makes the
trade-off explicit so accuracy targets come from arithmetic rather than optimism.

All times are seconds. The two parameters that matter most, keystroke time and chord
time, are properties of the user's hands and are measured by `scripts/chord_drill.py`
rather than assumed here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeedParams:
    """Timing assumptions behind the comparison."""

    keystroke: float = 0.20  # seconds per keystroke, 0.20 is about 60 wpm
    chord: float = 0.60  # seconds to form and release one chord
    notice: float = 0.40  # seconds to notice a wrong word once it appears
    cycle: float = 0.30  # seconds to press the cycle key and re-read the result


def typing_cost(length: int, params: SpeedParams) -> float:
    """Cost of typing a word normally, including the space after it."""
    return (length + 1) * params.keystroke


def chording_cost(length: int, top1: float, top3: float, params: SpeedParams) -> float:
    """Expected cost of chording a word of `length` letters.

    Three outcomes: the top word is right, or it is inside the top three and one or two
    cycle presses fix it, or it is not there at all and the word is undone and typed out.
    """
    correct = params.chord
    in_top3 = params.chord + params.notice + 1.5 * params.cycle
    missed = params.chord + params.notice + params.keystroke + typing_cost(length, params)
    return top1 * correct + (top3 - top1) * in_top3 + (1 - top3) * missed


def break_even_top1(length: int, top3: float, params: SpeedParams) -> float:
    """Smallest top-1 accuracy at which chording beats typing, given a top-3 rate.

    Solved directly: cost is linear in top-1 once top-3 is fixed.
    """
    correct = params.chord
    in_top3 = params.chord + params.notice + 1.5 * params.cycle
    missed = params.chord + params.notice + params.keystroke + typing_cost(length, params)
    target = typing_cost(length, params)
    # cost(p) = p*correct + (top3-p)*in_top3 + (1-top3)*missed
    slope = correct - in_top3
    constant = top3 * in_top3 + (1 - top3) * missed
    if slope == 0:
        return 0.0 if constant <= target else 1.0
    return max(0.0, min(1.0, (target - constant) / slope))


def render(params: SpeedParams, top1: float, top3: float, lengths=(4, 5, 6, 7, 9, 12)) -> str:
    lines = [
        "== is chording faster? ==",
        f"  keystroke {params.keystroke * 1000:.0f} ms, chord {params.chord * 1000:.0f} ms, "
        f"notice {params.notice * 1000:.0f} ms, cycle {params.cycle * 1000:.0f} ms",
        f"  measured accuracy: top1 {top1:.1%}, top3 {top3:.1%}",
        "",
        "  length   typed   chorded   change   top1 needed to break even",
    ]
    for length in lengths:
        typed = typing_cost(length, params)
        chorded = chording_cost(length, top1, top3, params)
        change = (typed - chorded) / typed
        needed = break_even_top1(length, top3, params)
        lines.append(
            f"  {length:>6d} {typed:>7.2f}s {chorded:>8.2f}s {change:>8.1%} {needed:>26.1%}"
        )
    return "\n".join(lines)
