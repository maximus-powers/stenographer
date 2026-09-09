"""Frequency-only ranking: the untrained baseline and the tap's fallback.

This predictor ignores context entirely and returns the most frequent words containing
the chord letters. It exists for two reasons: it is the number every trained model has to
beat to justify itself, and the same logic (as `FallbackIndex`) is what the tap process
falls back to when the brain misses its deadline.
"""

from __future__ import annotations

from pathlib import Path

from model.candidates import CandidateIndex
from model.vocab import Vocab
from stenographer.brain.predictor.base import Request


class UnigramPredictor:
    """Ranks candidates by corpus frequency alone."""

    name = "unigram"

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab
        self.index = CandidateIndex(vocab)

    @classmethod
    def load(cls, directory: Path = Path("data/vocab")) -> UnigramPredictor:
        return cls(Vocab.load(directory))

    def predict(self, request: Request, k: int = 5) -> list[str]:
        return self.index.top_k(request.letters, k)

    def warmup(self) -> None:
        self.predict(Request(letters="eht"), k=5)
