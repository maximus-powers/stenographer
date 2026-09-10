"""The chord model: a pretrained encoder with a whole-word output head.

The task is to name one word given the text before it and a set of letters that word
contains. That is close to what a masked language model already does, so the encoder
starts from pretrained BERT weights and only the output layer is new.

The head predicts a whole word in one step rather than assembling wordpieces. That
matters for two reasons. Candidate sets run to thousands of words, and a single masked
softmax over the vocabulary scores all of them in one matmul, whereas per-candidate
scoring cannot fit the latency budget. It also makes the candidate mask exact: a word the
chord cannot have produced is removed before the softmax, so it can never be emitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

# Chord letters ride in the second segment, after the context. Single characters are
# already wordpieces in every BERT vocabulary, so no tokenizer surgery is needed.
MASK_SEGMENT = 0
CHORD_SEGMENT = 1


@dataclass
class ModelConfig:
    encoder: str = "distilbert-base-uncased"
    vocab_size: int = 50_000
    max_length: int = 80
    left_window: int = 48
    right_window: int = 16
    # Probability that a training example carries any following text. Zero, because the
    # daemon never has any: it is built from what the tap reported typing, which is text
    # behind the caret, and reading ahead would need the Accessibility API work that is
    # not built. A quarter of examples used to carry right context the deployed model
    # could never be given. Raise this only if something starts supplying it.
    right_context_prob: float = 0.0
    # Probability that an example has its left context cut to a random shorter length,
    # matching a daemon that has only seen part of a session. See
    # `model.corpus.trim_to_runtime_context`.
    short_context_prob: float = 0.35
    dropout: float = 0.1


class ChordModel(nn.Module):
    """Encoder plus a linear head over whole words."""

    def __init__(self, config: ModelConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(config.encoder)
        self.encoder = encoder
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(hidden, config.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        mask_position: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        """Word logits at the masked slot, shape (batch, vocab_size)."""
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self.accepts_token_types:
            kwargs["token_type_ids"] = token_type_ids
        hidden_states = self.encoder(**kwargs).last_hidden_state
        pooled = hidden_states[torch.arange(hidden_states.size(0)), mask_position]
        return self.head(self.dropout(pooled))

    @property
    def accepts_token_types(self) -> bool:
        # DistilBERT has no segment embeddings; BERT does. Cached because it is consulted
        # on every forward pass.
        if not hasattr(self, "_accepts_token_types"):
            names = self.encoder.forward.__code__.co_varnames
            object.__setattr__(self, "_accepts_token_types", "token_type_ids" in names)
        return self._accepts_token_types  # type: ignore[attr-defined]

    @torch.no_grad()
    def seed_head_from_embeddings(self, words: list[str], tokenizer) -> int:
        """Initialise each output row from the mean of that word's input embeddings.

        A fresh head starts with no idea that its row for "because" has anything to do
        with the token "because". Averaging the word's own wordpiece embeddings gives
        every row a meaningful starting direction, which is worth doing because the head
        is the largest new parameter block in the model.
        """
        embeddings = self.encoder.get_input_embeddings().weight
        hidden = embeddings.size(1)
        if self.head.weight.size(1) != hidden:
            raise ValueError("head width does not match encoder hidden size")

        seeded = 0
        rows = torch.zeros(len(words), hidden)
        for position, word in enumerate(words):
            pieces = tokenizer(word, add_special_tokens=False)["input_ids"]
            if not pieces:
                continue
            rows[position] = embeddings[torch.tensor(pieces)].mean(dim=0)
            seeded += 1
        # Match the scale the head would have had under its default initialisation, so the
        # seeding changes direction without blowing up the initial logits.
        norm = rows.norm(dim=1, keepdim=True).clamp(min=1e-6)
        rows = rows / norm * (hidden ** -0.5)
        self.head.weight.copy_(rows)
        self.head.bias.zero_()
        return seeded


def masked_top_k(
    logits: Tensor, candidate_mask: Tensor, k: int = 5
) -> tuple[Tensor, Tensor]:
    """Top-k words restricted to the candidate set.

    Words outside the set are pushed to negative infinity rather than merely
    down-weighted, so a word the chord could not have produced is never emitted.
    """
    masked = logits.masked_fill(~candidate_mask, float("-inf"))
    k = min(k, int(candidate_mask.sum(dim=-1).max().item()) or 1)
    return masked.topk(k, dim=-1)
