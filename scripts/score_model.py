"""Score a trained checkpoint against the frequency baseline, on the same trials.

The baseline and the model see identical chords over identical held-out text, so the
difference between them is the value context adds and nothing else. That comparison is
the go/no-go for the whole project: the speed model says roughly what top-1 accuracy is
needed before chording beats typing, and this says whether the model reaches it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model import corpus, speed_model
from model import eval as harness
from model.synth import SynthConfig
from model.vocab import Vocab
from stenographer.brain.predictor.unigram import UnigramPredictor


class BatchedScorer:
    """Runs the harness over a predictor that prefers batches."""

    def __init__(self, predictor, batch_size: int = 128) -> None:
        self.predictor = predictor
        self.batch_size = batch_size

    def report(self, trials, name: str, k: int = 5) -> harness.Report:
        report = harness.Report(name=name)
        for start in range(0, len(trials), self.batch_size):
            chunk = trials[start : start + self.batch_size]
            ranked = self.predictor.predict_batch([t.request for t in chunk], k)
            for trial, words in zip(chunk, ranked):
                lowered = [w.lower() for w in words]
                rank = lowered.index(trial.target) if trial.target in lowered else None
                report.add(trial, rank)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/distilbert-6m/checkpoint.pt"))
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab"))
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--trials", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chord-time", type=float, default=0.60)
    parser.add_argument("--keystroke-time", type=float, default=0.20)
    parser.add_argument("--cpu", action="store_true", help="score on the CPU instead of MPS")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint}; train one first")

    vocab = Vocab.load(args.vocab)
    trials, skipped = harness.build_trials(
        corpus.iter_examples(corpus.iter_lines(args.corpus, split=args.split)),
        vocab,
        limit=args.trials,
        config=SynthConfig(),
    )
    chordable = skipped["chorded"] / max(1, skipped["tokens seen"])
    print(
        f"{len(trials):,} trials from {args.corpus}:{args.split} "
        f"({chordable:.1%} of tokens were chordable)\n"
    )

    baseline = harness.score(
        UnigramPredictor(vocab), trials, name="baseline: frequency only, no context"
    )
    print(baseline.render(), "\n")

    from stenographer.brain.predictor.bert import BertPredictor

    device = torch.device("cpu") if args.cpu else None
    predictor = BertPredictor.load(args.checkpoint, args.vocab, device=device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint from step {payload.get('step', '?'):,} on {predictor.device}")
    predictor.warmup()

    scorer = BatchedScorer(predictor, args.batch_size)
    trained = scorer.report(trials, name=f"trained model ({predictor.encoder.config.encoder})")
    print(trained.render(), "\n")

    base1, base3, base5 = baseline.overall.rates()
    got1, got3, got5 = trained.overall.rates()
    print("== what context bought ==")
    print(f"  top1  {base1:6.1%} -> {got1:6.1%}   ({got1 - base1:+.1%})")
    print(f"  top3  {base3:6.1%} -> {got3:6.1%}   ({got3 - base3:+.1%})")
    print(f"  top5  {base5:6.1%} -> {got5:6.1%}   ({got5 - base5:+.1%})")

    print("\n== single-prediction latency ==")
    single = trials[0].request
    for _ in range(5):
        predictor.predict(single, k=5)
    timings = []
    for trial in trials[:60]:
        predictor.predict(trial.request, k=5)
        timings.append(predictor.last_timing)
    timings.sort(key=lambda t: t.total_ms)
    median = timings[len(timings) // 2]
    p90 = timings[int(len(timings) * 0.9)]
    print(
        f"  median  tokenise {median.tokenise_ms:5.1f} ms  forward {median.forward_ms:5.1f} ms  "
        f"rank {median.rank_ms:5.1f} ms  total {median.total_ms:5.1f} ms"
    )
    print(f"  p90     total {p90.total_ms:5.1f} ms   (the tap gives up at 50 ms)")

    params = speed_model.SpeedParams(keystroke=args.keystroke_time, chord=args.chord_time)
    print()
    print(speed_model.render(params, got1, got3))


if __name__ == "__main__":
    main()
