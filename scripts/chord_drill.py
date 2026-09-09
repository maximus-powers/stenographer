"""Measure how long you take to form a chord, and how accurately.

The speed model needs one number that cannot be derived from any corpus: how long your
hands take to press several letters at once and move on. Everything else about whether
chording beats typing follows from that, so it is worth measuring before building the
keyboard layer around it.

The drill shows a word, you press some of its letters together, and it times you. It also
runs a plain typing round so the comparison uses your own typing speed rather than an
assumed one. It reads the terminal directly and needs no Accessibility permission.

Nothing is written to disk unless you pass --save.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import termios
import time
import tty
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.speed_model import SpeedParams, break_even_top1, render
from model.synth import is_typable, same_finger_pairs
from model.vocab import Vocab, distinct_letters

# Words long enough to be worth chording, spread across lengths and shapes.
DEFAULT_WORDS = [
    "because", "there", "something", "different", "understand", "probably",
    "government", "important", "together", "children", "question", "problem",
    "morning", "tomorrow", "everything", "remember", "between", "another",
    "thought", "through", "should", "people", "really", "before",
]


@dataclass
class Attempt:
    word: str
    expected: str  # the letters the drill asked for
    pressed: str  # the letters actually received
    seconds: float
    correct: bool


class RawTerminal:
    """Read single keypresses without waiting for Enter.

    Falls back to ordinary buffered reads when stdin is not a terminal, so the drill can
    be exercised from a script. Timings collected that way are meaningless, and the
    summary says so.
    """

    def __init__(self) -> None:
        self.interactive = sys.stdin.isatty()
        self.fd = -1
        self.saved = None

    def __enter__(self) -> RawTerminal:
        if not self.interactive:
            return self
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_) -> None:
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def read(self, count: int = 1) -> str:
        character = sys.stdin.read(count)
        if not character:
            raise KeyboardInterrupt  # end of input
        return character


def prefix_chord(word: str, size: int) -> str:
    return distinct_letters(word)[:size]


def run_chord_round(words: list[str], size: int, terminal: RawTerminal) -> list[Attempt]:
    attempts: list[Attempt] = []
    for word in words:
        expected = prefix_chord(word, size)
        if len(expected) < size:
            continue
        hint = " ".join(sorted(expected))
        sys.stdout.write(f"\r\n  {word:<14s} press {{{hint}}} together: ")
        sys.stdout.flush()
        started = time.perf_counter()
        pressed = ""
        while len(pressed) < len(expected):
            character = terminal.read(1)
            if character in ("\x03", "\x04"):  # Ctrl-C, Ctrl-D
                raise KeyboardInterrupt
            if character == "\x1b":
                raise KeyboardInterrupt
            if character.isalpha():
                pressed += character.lower()
        elapsed = time.perf_counter() - started
        correct = set(pressed) == set(expected)
        attempts.append(
            Attempt(
                word=word,
                expected="".join(sorted(expected)),
                pressed="".join(sorted(set(pressed))),
                seconds=elapsed,
                correct=correct,
            )
        )
        sys.stdout.write(f"{'ok' if correct else 'got ' + pressed}  {elapsed * 1000:5.0f} ms")
        sys.stdout.flush()
    return attempts


def run_typing_round(words: list[str], terminal: RawTerminal) -> list[Attempt]:
    attempts: list[Attempt] = []
    for word in words:
        sys.stdout.write(f"\r\n  type {word:<14s} ")
        sys.stdout.flush()
        started = time.perf_counter()
        typed = ""
        while len(typed) < len(word):
            character = terminal.read(1)
            if character in ("\x03", "\x04", "\x1b"):
                raise KeyboardInterrupt
            if character.isalpha():
                typed += character.lower()
                sys.stdout.write(character)
                sys.stdout.flush()
        elapsed = time.perf_counter() - started
        attempts.append(
            Attempt(
                word=word,
                expected=word,
                pressed=typed,
                seconds=elapsed,
                correct=typed == word,
            )
        )
        sys.stdout.write(f"  {elapsed * 1000:5.0f} ms")
        sys.stdout.flush()
    return attempts


def summarise(attempts: list[Attempt], label: str) -> dict:
    if not attempts:
        return {}
    times = sorted(a.seconds for a in attempts)
    correct = [a for a in attempts if a.correct]
    letters = sum(len(a.word) for a in attempts)
    return {
        "label": label,
        "attempts": len(attempts),
        "accuracy": len(correct) / len(attempts),
        "median_seconds": statistics.median(times),
        "mean_seconds": statistics.fmean(times),
        "p90_seconds": times[min(len(times) - 1, int(len(times) * 0.9))],
        "letters": letters,
    }


def render_summary(summary: dict) -> str:
    if not summary:
        return "  (no attempts)"
    return (
        f"  {summary['label']:<22s} n={summary['attempts']:<4d} "
        f"median {summary['median_seconds'] * 1000:6.0f} ms   "
        f"p90 {summary['p90_seconds'] * 1000:6.0f} ms   "
        f"letters correct {summary['accuracy']:5.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12, help="words per round")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", type=Path, help="write the measurements to this file")
    parser.add_argument("--skip-typing", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    words = list(DEFAULT_WORDS)
    if args.vocab.exists():
        vocab = Vocab.load(args.vocab)
        pool = [w for w in vocab.words[:6000] if len(w) >= 6 and w.isalpha() and w.islower()]
        if len(pool) > args.rounds:
            words = rng.sample(pool, min(len(pool), args.rounds * (len(args.sizes) + 1)))
    rng.shuffle(words)

    print(__doc__.strip().splitlines()[0])
    print()
    if not sys.stdin.isatty():
        print("stdin is not a terminal, so the timings below are not real measurements.")
        print("Run this from a terminal to measure your own hands.")
    else:
        print("Press the letters shown at the same time, as you would a chord.")
        print("Escape or Ctrl-C stops. The terminal echoes nothing during chord rounds.")
    print()

    results: dict[str, dict] = {}
    raw_attempts: dict[str, list] = {}
    try:
        with RawTerminal() as terminal:
            cursor = 0
            for size in args.sizes:
                batch = words[cursor : cursor + args.rounds]
                cursor += args.rounds
                sys.stdout.write(f"\r\n-- {size}-letter chords --")
                sys.stdout.flush()
                attempts = run_chord_round(batch, size, terminal)
                results[f"chord-{size}"] = summarise(attempts, f"{size}-letter chords")
                raw_attempts[f"chord-{size}"] = [asdict(a) for a in attempts]

            if not args.skip_typing:
                batch = words[cursor : cursor + args.rounds]
                sys.stdout.write("\r\n\r\n-- plain typing, for comparison --")
                sys.stdout.flush()
                attempts = run_typing_round(batch, terminal)
                results["typing"] = summarise(attempts, "typed in full")
                raw_attempts["typing"] = [asdict(a) for a in attempts]
    except KeyboardInterrupt:
        print("\r\n\nstopped early")

    print("\n")
    print("== measurements ==")
    for summary in results.values():
        print(render_summary(summary))

    typing = results.get("typing")
    chord_summaries = [s for key, s in results.items() if key.startswith("chord") and s]
    if typing and chord_summaries:
        keystroke = typing["mean_seconds"] / max(
            1, typing["letters"] / typing["attempts"] + 1
        )
        best = min(chord_summaries, key=lambda s: s["median_seconds"])
        params = SpeedParams(keystroke=keystroke, chord=best["median_seconds"])
        print()
        print(
            f"  your keystroke time  {keystroke * 1000:.0f} ms "
            f"(about {12.0 / keystroke / 5:.0f} words per minute)"
        )
        print(f"  your best chord time {best['median_seconds'] * 1000:.0f} ms ({best['label']})")
        print()
        print(render(params, 0.80, 0.92))
        print()
        needed = break_even_top1(6, 0.92, params)
        print(
            f"  With your timings, a six-letter word needs top-1 accuracy of "
            f"{needed:.1%} before chording wins."
        )
        results["derived"] = {
            "keystroke_seconds": keystroke,
            "chord_seconds": best["median_seconds"],
            "break_even_top1_len6": needed,
        }

    awkward = [
        (size, prefix_chord(word, size))
        for size in args.sizes
        for word in words[: args.rounds]
        if len(distinct_letters(word)) >= size
    ]
    same_finger = [c for _, c in awkward if not is_typable(c)]
    if awkward:
        print()
        print(
            f"  {len(same_finger)}/{len(awkward)} of the chords drilled put two letters "
            f"on one finger"
        )
        if same_finger:
            worst = max(same_finger, key=same_finger_pairs)
            print(f"  the most awkward was {{{','.join(sorted(worst))}}}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({"summary": results, "attempts": raw_attempts}, indent=2))
        print(f"\n  saved to {args.save}")


if __name__ == "__main__":
    main()
