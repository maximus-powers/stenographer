"""Materialise a public corpus to local text files.

Training reads the corpus tens of times over, and the Hugging Face cache fetches large
splits lazily in chunks: an early run spent 46 minutes on 300 steps at 5% CPU, waiting on
the network rather than computing. Writing the text out once removes that entirely, and
makes a run reproducible without a network connection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--config", default="wikitext-103-raw-v1")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--out", type=Path, default=Path("data/corpus"))
    args = parser.parse_args()

    from datasets import load_dataset

    args.out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        destination = args.out / f"{split}.txt"
        if destination.exists():
            print(f"{split}: already at {destination} ({destination.stat().st_size / 1e6:.0f} MB)")
            continue
        print(f"{split}: downloading ...", flush=True)
        started = time.time()
        dataset = load_dataset(args.dataset, args.config, split=split)
        lines = words = 0
        temporary = destination.with_suffix(".partial")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in dataset:
                text = row["text"].strip()
                if not text:
                    continue
                handle.write(text + "\n")
                lines += 1
                words += text.count(" ") + 1
        temporary.rename(destination)
        print(
            f"{split}: {lines:,} lines, {words:,} words -> {destination} "
            f"({destination.stat().st_size / 1e6:.0f} MB, {time.time() - started:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
