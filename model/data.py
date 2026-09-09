"""Training data: public corpus text plus synthetic chords, assembled into batches.

Each example asks the model one question. Given the words before (and sometimes after) a
gap, and a set of letters the missing word contains, which word is it?

Examples are built on the fly rather than written to disk. Chords are sampled fresh every
epoch, so the model sees many different briefs for the same word instead of memorising
one, and no dataset needs materialising.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from model.corpus import Example, iter_examples, iter_lines
from model.net import ModelConfig
from model.synth import SynthConfig, sample_chord
from model.vocab import Vocab, distinct_letters


@dataclass
class Batch:
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    mask_position: Tensor
    target: Tensor
    candidate_mask: Tensor | None = None

    def to(self, device: torch.device) -> Batch:
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            token_type_ids=self.token_type_ids.to(device),
            mask_position=self.mask_position.to(device),
            target=self.target.to(device),
            candidate_mask=None if self.candidate_mask is None else self.candidate_mask.to(device),
        )

    def __len__(self) -> int:
        return int(self.input_ids.size(0))


class Encoder:
    """Turns an Example plus a chord into model input ids.

    Wordpieces for each distinct word are cached, because a corpus reuses its vocabulary
    heavily and tokenising "the" a million times is wasted work.
    """

    def __init__(self, tokenizer, config: ModelConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self._cache: dict[str, list[int]] = {}
        self._letters = {
            letter: tokenizer(letter, add_special_tokens=False)["input_ids"]
            for letter in "abcdefghijklmnopqrstuvwxyz"
        }
        missing = [letter for letter, ids in self._letters.items() if len(ids) != 1]
        if missing:
            raise ValueError(
                f"tokenizer splits single letters {missing}; chord letters must be one token each"
            )

    def pieces(self, token: str) -> list[int]:
        cached = self._cache.get(token)
        if cached is None:
            cached = self.tokenizer(token, add_special_tokens=False)["input_ids"]
            self._cache[token] = cached
        return cached

    def context_ids(self, tokens: Sequence[str], limit: int, keep_tail: bool) -> list[int]:
        """Wordpieces for a run of tokens, trimmed to `limit`.

        `keep_tail` keeps the words nearest the gap, which is what left context needs;
        right context keeps its head for the same reason.
        """
        ids: list[int] = []
        ordered = reversed(tokens) if keep_tail else tokens
        for token in ordered:
            piece = self.pieces(token)
            if len(ids) + len(piece) > limit:
                break
            if keep_tail:
                ids = piece + ids
            else:
                ids.extend(piece)
        return ids

    def encode(
        self, example: Example, chord: str, *, include_right: bool
    ) -> tuple[list[int], list[int], int]:
        """Build (input_ids, token_type_ids, mask_position) for one example."""
        left = self.context_ids(example.left, self.config.left_window, keep_tail=True)
        right = (
            self.context_ids(example.right, self.config.right_window, keep_tail=False)
            if include_right
            else []
        )
        chord_ids = [self._letters[c][0] for c in sorted(set(chord))]

        # Reserve room for [CLS], [MASK], both [SEP]s and the chord before trimming context.
        budget = self.config.max_length - 4 - len(chord_ids)
        while len(left) + len(right) > budget:
            if len(right) >= len(left) and right:
                right.pop()
            elif left:
                left.pop(0)
            else:
                break

        ids = [self.cls_id] + left
        mask_position = len(ids)
        ids += [self.mask_id] + right + [self.sep_id]
        types = [0] * len(ids)
        ids += chord_ids + [self.sep_id]
        types += [1] * (len(chord_ids) + 1)
        return ids, types, mask_position

    def collate(self, rows: Sequence[tuple[list[int], list[int], int, int]]) -> Batch:
        width = max(len(ids) for ids, _, _, _ in rows)
        size = len(rows)
        input_ids = np.full((size, width), self.pad_id, dtype=np.int64)
        token_types = np.zeros((size, width), dtype=np.int64)
        attention = np.zeros((size, width), dtype=np.int64)
        positions = np.zeros(size, dtype=np.int64)
        targets = np.zeros(size, dtype=np.int64)

        for row, (ids, types, position, target) in enumerate(rows):
            input_ids[row, : len(ids)] = ids
            token_types[row, : len(types)] = types
            attention[row, : len(ids)] = 1
            positions[row] = position
            targets[row] = target

        return Batch(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention),
            token_type_ids=torch.from_numpy(token_types),
            mask_position=torch.from_numpy(positions),
            target=torch.from_numpy(targets),
        )


class ExampleStream:
    """Yields encoded training rows from a corpus, cycling for as long as asked.

    Very frequent words are subsampled the way word2vec does it. Without that, a quarter
    of all training signal would be spent on a handful of function words the frequency
    baseline already gets right, and rare words would barely be seen.
    """

    def __init__(
        self,
        vocab: Vocab,
        encoder: Encoder,
        *,
        corpus: str = "wikitext-103-raw-v1",
        split: str = "train",
        synth: SynthConfig | None = None,
        subsample: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.vocab = vocab
        self.encoder = encoder
        self.corpus = corpus
        self.split = split
        self.synth = synth or SynthConfig()
        self.subsample = subsample
        self.rng = random.Random(seed)
        self.keep_prob = self._keep_probabilities()

    def _keep_probabilities(self) -> np.ndarray:
        """Per-word probability of using an occurrence as a training target."""
        zipf = self.vocab.freqs.astype(np.float64)
        share = np.power(10.0, zipf - 9.0)  # Zipf value back to a per-token share
        share = np.clip(share, 1e-12, 1.0)
        if self.subsample <= 0:
            return np.ones_like(share)
        keep = (np.sqrt(share / self.subsample) + 1.0) * (self.subsample / share)
        return np.clip(keep, 0.0, 1.0)

    def rows(self, count: int) -> Iterator[tuple[list[int], list[int], int, int]]:
        produced = 0
        while produced < count:
            lines = iter_lines(self.corpus, split=self.split)
            for example in iter_examples(
                lines,
                left_window=self.encoder.config.left_window,
                right_window=self.encoder.config.right_window,
            ):
                lower = example.target.lower()
                target = self.vocab.rank(lower)
                if target is None or len(distinct_letters(lower)) < 2:
                    continue
                if self.rng.random() > self.keep_prob[target]:
                    continue
                drawn = sample_chord(lower, self.rng, self.synth)
                if drawn is None:
                    continue
                chord, _ = drawn
                include_right = self.rng.random() < self.encoder.config.right_context_prob
                ids, types, position = self.encoder.encode(
                    example, chord, include_right=include_right
                )
                yield ids, types, position, target
                produced += 1
                if produced >= count:
                    return

    def batches(self, batch_size: int, steps: int) -> Iterator[Batch]:
        pending: list[tuple[list[int], list[int], int, int]] = []
        for row in self.rows(batch_size * steps):
            pending.append(row)
            if len(pending) == batch_size:
                yield self.encoder.collate(pending)
                pending = []
        if pending:
            yield self.encoder.collate(pending)

    def prefetched_batches(
        self, batch_size: int, steps: int, depth: int = 4
    ) -> Iterator[Batch]:
        """Same batches, built on a worker thread so data and compute overlap.

        Assembling a batch and running one is otherwise serial in a single thread, which
        costs roughly a quarter of training throughput. Tokenisation releases the GIL
        often enough that one worker is enough to hide it.
        """
        import queue
        import threading

        finished = object()
        pipe: queue.Queue = queue.Queue(maxsize=depth)

        def fill() -> None:
            try:
                for batch in self.batches(batch_size, steps):
                    pipe.put(batch)
            except Exception as error:  # surface it on the consumer side
                pipe.put(error)
            finally:
                pipe.put(finished)

        worker = threading.Thread(target=fill, name="example-stream", daemon=True)
        worker.start()
        while True:
            item = pipe.get()
            if item is finished:
                return
            if isinstance(item, Exception):
                raise item
            yield item


def build_eval_batches(
    examples: Iterable[Example],
    vocab: Vocab,
    encoder: Encoder,
    *,
    limit: int,
    batch_size: int,
    seed: int = 0,
    synth: SynthConfig | None = None,
) -> tuple[list[Batch], list[str]]:
    """Fixed batches for validation, so successive checkpoints are compared on the same data."""
    rng = random.Random(seed)
    synth = synth or SynthConfig()
    rows: list[tuple[list[int], list[int], int, int]] = []
    chords: list[str] = []
    for example in examples:
        lower = example.target.lower()
        target = vocab.rank(lower)
        if target is None or len(distinct_letters(lower)) < 2:
            continue
        drawn = sample_chord(lower, rng, synth)
        if drawn is None:
            continue
        chord, _ = drawn
        ids, types, position = encoder.encode(example, chord, include_right=False)
        rows.append((ids, types, position, target))
        chords.append(chord)
        if len(rows) >= limit:
            break

    batches = [
        encoder.collate(rows[start : start + batch_size])
        for start in range(0, len(rows), batch_size)
    ]
    return batches, chords
