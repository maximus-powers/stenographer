"""Run the prediction daemon.

Start this before `scripts/steno.py`. It loads the model, which takes a few seconds, and
then answers chord lookups over a Unix socket. Stopping it does not stop typing: the tap
falls back to its own frequency index.

    python scripts/stenod.py --checkpoint runs/distilbert-6m/checkpoint.pt

With no checkpoint it serves frequency-only predictions, which is useful for exercising
the plumbing before a model has finished training.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.vocab import Vocab
from stenographer.brain.predictor.unigram import UnigramPredictor
from stenographer.brain.server import Brain, BrainServer
from stenographer.protocol import socket_path


def load_predictor(checkpoint: Path | None, vocab_dir: Path, cpu: bool):
    vocab = Vocab.load(vocab_dir)
    if checkpoint is None or not checkpoint.exists():
        if checkpoint is not None:
            print(f"no checkpoint at {checkpoint}", flush=True)
        print("serving frequency-only predictions (no context)", flush=True)
        return UnigramPredictor(vocab)

    import torch

    from stenographer.brain.predictor.bert import BertPredictor

    device = torch.device("cpu") if cpu else None
    print(f"loading {checkpoint} ...", flush=True)
    predictor = BertPredictor.load(checkpoint, vocab_dir, device=device)
    print(f"  {predictor.encoder.config.encoder} on {predictor.device}", flush=True)
    predictor.warmup()
    print("  warmed up", flush=True)
    return predictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # The checkpoint ./retrain produces. The older runs/distilbert-6m was trained against
    # a candidate rule this no longer uses - 65% of its examples were briefs that do not
    # resolve now - so pointing at it silently serves worse predictions than no model.
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/prefix-40000/checkpoint.pt")
    )
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab"))
    parser.add_argument("--socket", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-model", action="store_true", help="frequency only")
    parser.add_argument("--verbose", action="store_true", help="log every prediction")
    args = parser.parse_args()

    predictor = load_predictor(
        None if args.no_model else args.checkpoint, args.vocab, args.cpu
    )
    brain = Brain(predictor=predictor)
    server = BrainServer(brain, path=args.socket or socket_path(), verbose=args.verbose)

    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        path = args.socket or socket_path()
        if path.exists():
            path.unlink()
        if brain.stats.requests:
            print(
                f"served {brain.stats.requests:,} predictions, "
                f"mean {brain.stats.mean_ms:.1f} ms, slowest {brain.stats.slowest_ms:.1f} ms, "
                f"{brain.stats.over_deadline} over the {brain.deadline_ms:.0f} ms deadline"
            )


if __name__ == "__main__":
    main()
