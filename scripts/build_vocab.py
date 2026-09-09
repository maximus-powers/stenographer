"""Build and save the shared vocabulary. Run once before anything else."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import corpus
from model import vocab as vocab_mod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=vocab_mod.DEFAULT_SIZE)
    parser.add_argument("--out", type=Path, default=Path("data/vocab"))
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--restrict",
        action="store_true",
        help="keep only words the corpus contains (needs a large corpus)",
    )
    args = parser.parse_args()

    print(f"reading {args.corpus}:{args.split} for casing statistics ...", flush=True)
    counts, capitalised = corpus.count_words(corpus.iter_lines(args.corpus, split=args.split))
    print(f"  {sum(counts.values()):,} word tokens, {len(counts):,} types")

    print(f"building top-{args.size:,} vocabulary from wordfreq ...", flush=True)
    built = vocab_mod.build(
        args.size, counts=counts, capitalised=capitalised, restrict=args.restrict
    )
    built.save(args.out)

    capitalised_words = [w for w in built.words if w[0].isupper()]
    print(f"  {len(built):,} words -> {args.out}")
    print(f"  {len(capitalised_words):,} carry canonical capitals, e.g. {capitalised_words[:12]}")
    print(f"  most frequent: {built.words[:12]}")
    print(f"  least frequent: {built.words[-6:]}")


if __name__ == "__main__":
    main()
