"""Ranking candidates with the trained model.

This is the inference half of `model/train.py`, and it exists separately because the two
have different priorities. Training cares about throughput over millions of examples;
this cares about the latency of a single prediction, because the user is waiting for a
word to appear and the tap gives up after 50 milliseconds.

The candidate mask is what makes a single forward pass enough. Scoring thousands of
candidate words one at a time could never fit the budget, but restricting one softmax
over the whole vocabulary costs nothing extra and rules out every word the chord could
not have produced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from model.candidates import CandidateIndex
from model.corpus import Example
from model.data import Encoder
from model.net import ChordModel, ModelConfig, masked_top_k
from model.vocab import Vocab
from stenographer.brain.predictor.base import Request


@dataclass
class Timing:
    """Where the time went, so the latency budget can be checked against reality."""

    tokenise_ms: float = 0.0
    forward_ms: float = 0.0
    rank_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.tokenise_ms + self.forward_ms + self.rank_ms


class BertPredictor:
    """Scores the words a chord could have produced, using the context around it."""

    name = "bert"

    def __init__(
        self,
        model: ChordModel,
        vocab: Vocab,
        encoder: Encoder,
        device: torch.device | None = None,
    ) -> None:
        from model.train import pick_device

        self.vocab = vocab
        self.encoder = encoder
        self.index = CandidateIndex(vocab)
        # Default to the fastest device available rather than the CPU. Getting this wrong
        # is quiet and expensive: a forward pass costs about 34 ms on the CPU here against
        # a 50 ms deadline, so it works and merely wastes most of the budget.
        self.device = device or pick_device()
        self.model = model.to(self.device).eval()
        self.last_timing = Timing()

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        vocab_dir: Path = Path("data/vocab"),
        device: torch.device | None = None,
    ) -> BertPredictor:
        from transformers import AutoModel, AutoTokenizer

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = ModelConfig(**payload["model_config"])
        vocab = Vocab.load(vocab_dir)
        if config.vocab_size != len(vocab):
            raise ValueError(
                f"checkpoint expects {config.vocab_size:,} words but the vocabulary at "
                f"{vocab_dir} has {len(vocab):,}"
            )

        tokenizer = AutoTokenizer.from_pretrained(config.encoder)
        # Build the encoder from the pretrained config without downloading weights we are
        # about to overwrite from the checkpoint.
        backbone = AutoModel.from_pretrained(config.encoder)
        model = ChordModel(config, encoder=backbone)
        model.load_state_dict(payload["model"])
        return cls(model, vocab, Encoder(tokenizer, config), device=device)

    def warmup(self, rounds: int = 3) -> None:
        """Run a few predictions so shader compilation does not land on the user."""
        request = Request(letters="eht", left=("the", "cat", "sat", "on"))
        for _ in range(rounds):
            self.predict(request, k=5)

    @torch.inference_mode()
    def predict(self, request: Request, k: int = 5) -> list[str]:
        return self.predict_batch([request], k)[0]

    @torch.inference_mode()
    def predict_batch(self, requests: list[Request], k: int = 5) -> list[list[str]]:
        """Rank several chords at once, which is how the offline harness runs."""
        if not requests:
            return []
        timing = Timing()

        started = time.perf_counter()
        rows = []
        masks = np.zeros((len(requests), len(self.vocab)), dtype=bool)
        for position, request in enumerate(requests):
            example = Example(
                target="",
                left=list(request.left),
                right=list(request.right),
                sentence_start=request.sentence_start,
            )
            ids, types, mask_position = self.encoder.encode(
                example, request.letters, include_right=bool(request.right)
            )
            rows.append((ids, types, mask_position, 0))
            candidate_mask = self.index.mask_for(request.letters)
            if not candidate_mask.any():
                # Nothing contains every chord letter, so allow one letter to be wrong
                # rather than returning nothing at all.
                relaxed = self.index.relaxed_candidates(request.letters)
                candidate_mask[relaxed] = True
            masks[position] = candidate_mask
        batch = self.encoder.collate(rows)
        timing.tokenise_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        moved = batch.to(self.device)
        logits = self.model(
            moved.input_ids, moved.attention_mask, moved.mask_position, moved.token_type_ids
        )
        if self.device.type == "mps":
            torch.mps.synchronize()
        timing.forward_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        candidate_mask = torch.from_numpy(masks).to(self.device)
        _, ranked = masked_top_k(logits.float(), candidate_mask, k=k)
        ranked_cpu = ranked.cpu().numpy()
        results: list[list[str]] = []
        for position in range(len(requests)):
            allowed = int(masks[position].sum())
            words = [
                self.vocab.words[index]
                for index in ranked_cpu[position][: min(k, allowed)]
                if masks[position][index]
            ]
            results.append(words)
        timing.rank_ms = (time.perf_counter() - started) * 1000

        self.last_timing = timing
        return results
