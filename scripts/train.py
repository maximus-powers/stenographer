"""Fine-tune the chord model. Start with --steps 300 to check the pipeline end to end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.net import ModelConfig
from model.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder", default="distilbert-base-uncased")
    parser.add_argument("--out", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab"))
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-examples", type=int, default=2_000)
    parser.add_argument("--encoder-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    train(
        vocab_dir=args.vocab,
        out_dir=args.out,
        model_config=ModelConfig(encoder=args.encoder),
        corpus=args.corpus,
        train_config=TrainConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
            eval_examples=args.eval_examples,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            warmup_steps=max(20, args.steps // 20),
            amp=not args.no_amp,
        ),
    )


if __name__ == "__main__":
    main()
