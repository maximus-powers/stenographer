"""Produce the milestone-zero report: ambiguity, baseline accuracy, and the speed verdict.

Run after `scripts/build_vocab.py`. Everything here is offline and needs no macOS
permissions, which is deliberate: it answers whether chorded typing can pay off before
any keyboard code is written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import ambiguity, corpus, speed_model
from model import eval as harness
from model.candidates import CandidateIndex
from model.synth import SynthConfig
from model.vocab import Vocab
from stenographer.brain.predictor.unigram import UnigramPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab"))
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chord-time", type=float, default=0.60)
    parser.add_argument("--keystroke-time", type=float, default=0.20)
    args = parser.parse_args()

    vocab = Vocab.load(args.vocab)
    index = CandidateIndex(vocab)
    print(f"vocabulary: {len(vocab):,} words from {args.vocab}\n")

    examples = corpus.iter_examples(corpus.iter_lines(args.corpus, split=args.split))
    trials, skipped = harness.build_trials(
        examples, vocab, limit=args.trials, seed=args.seed, config=SynthConfig()
    )

    seen = skipped["tokens seen"]
    print("== coverage: which words can be chorded at all ==")
    for reason in (
        "chorded",
        "too short to chord",
        "fewer than 2 distinct letters",
        "out of vocabulary",
    ):
        count = skipped[reason]
        if count:
            print(f"  {reason:<32s} {count:>8,}  {count / seen:6.1%}")
    print(f"  {'tokens seen':<32s} {seen:>8,}\n")

    print(ambiguity.render(ambiguity.measure(trials, index)), "\n")
    print(ambiguity.worst_chords(trials, index), "\n")

    predictor = UnigramPredictor(vocab)
    report = harness.score(predictor, trials, name="baseline: frequency only, no context")
    print(report.render(), "\n")

    print("== accuracy per chord size under one fixed briefing habit ==")
    print("  (prefix policy, noise off, so sizes are comparable)")
    clean = SynthConfig(substitution_noise=0.0, typability_resample=0.0)
    for size in (2, 3, 4, 5):
        sized, _ = harness.build_trials(
            corpus.iter_examples(corpus.iter_lines(args.corpus, split=args.split)),
            vocab,
            limit=args.trials // 2,
            seed=args.seed,
            config=clean,
            policy="prefix",
            size=size,
        )
        sized = [t for t in sized if t.size == size]
        if not sized:
            continue
        sized_report = harness.score(predictor, sized, name=f"prefix-{size}")
        a, b, c = sized_report.overall.rates()
        print(
            f"  size {size}  n={sized_report.overall.trials:>6,}  "
            f"top1 {a:6.1%}  top3 {b:6.1%}  top5 {c:6.1%}"
        )
    print()

    top1, top3, _ = report.overall.rates()
    params = speed_model.SpeedParams(keystroke=args.keystroke_time, chord=args.chord_time)
    print(speed_model.render(params, top1, top3))
    print(
        "\n  The chord-time figure above is an assumption until scripts/chord_drill.py\n"
        "  measures it. Break-even is what the trained model has to clear."
    )


if __name__ == "__main__":
    main()
