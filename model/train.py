"""Fine-tune the chord model.

Only the output head is new; the encoder starts from pretrained weights and is nudged
rather than retrained, which is why the head gets a much larger learning rate.

Validation always applies the candidate mask, because that is what inference does. An
unmasked accuracy number would flatter the model by crediting it for words the chord
could not have produced.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model.candidates import CandidateIndex
from model.data import Batch, Encoder, ExampleStream, build_eval_batches
from model.net import ChordModel, ModelConfig, masked_top_k
from model.synth import SynthConfig
from model.vocab import Vocab


@dataclass
class TrainConfig:
    steps: int = 4_000
    batch_size: int = 64
    encoder_lr: float = 5e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 200
    eval_every: int = 500
    eval_examples: int = 2_000
    log_every: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    amp: bool = True


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def candidate_masks(chords: list[str], index: CandidateIndex, batches: list[Batch]) -> None:
    """Attach the candidate mask each batch's rows are restricted to at inference."""
    offset = 0
    for batch in batches:
        rows = len(batch)
        mask = np.zeros((rows, len(index.vocab)), dtype=bool)
        for row in range(rows):
            mask[row] = index.mask_for(chords[offset + row])
        batch.candidate_mask = torch.from_numpy(mask)
        offset += rows


@torch.no_grad()
def evaluate(model: ChordModel, batches: list[Batch], device: torch.device) -> dict[str, float]:
    model.eval()
    seen = hits1 = hits3 = hits5 = 0
    for batch in batches:
        moved = batch.to(device)
        logits = model(
            moved.input_ids, moved.attention_mask, moved.mask_position, moved.token_type_ids
        )
        assert moved.candidate_mask is not None
        _, ranked = masked_top_k(logits.float(), moved.candidate_mask, k=5)
        target = moved.target.unsqueeze(-1)
        correct = ranked == target
        seen += int(correct.size(0))
        hits1 += int(correct[:, :1].any(dim=-1).sum())
        hits3 += int(correct[:, :3].any(dim=-1).sum())
        hits5 += int(correct[:, :5].any(dim=-1).sum())
    model.train()
    if not seen:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "n": 0}
    return {
        "top1": hits1 / seen,
        "top3": hits3 / seen,
        "top5": hits5 / seen,
        "n": float(seen),
    }


def learning_rate_scale(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return (step + 1) / max(1, config.warmup_steps)
    span = max(1, config.steps - config.warmup_steps)
    progress = (step - config.warmup_steps) / span
    return max(0.0, 1.0 - progress)


def train(
    vocab_dir: Path = Path("data/vocab"),
    out_dir: Path = Path("runs/pilot"),
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    synth: SynthConfig | None = None,
    corpus: str = "data/corpus",
) -> dict:
    from transformers import AutoTokenizer

    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()
    synth = synth or SynthConfig()
    torch.manual_seed(train_config.seed)

    vocab = Vocab.load(vocab_dir)
    model_config.vocab_size = len(vocab)
    index = CandidateIndex(vocab)
    device = pick_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device {device}, encoder {model_config.encoder}, vocab {len(vocab):,}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_config.encoder)
    encoder = Encoder(tokenizer, model_config)
    model = ChordModel(model_config)
    seeded = model.seed_head_from_embeddings(vocab.words, tokenizer)
    print(f"seeded {seeded:,} head rows from wordpiece embeddings", flush=True)
    model.to(device)

    print("building fixed validation set ...", flush=True)
    from model.corpus import iter_examples, iter_lines

    eval_batches, eval_chords = build_eval_batches(
        iter_examples(iter_lines(corpus, split="validation")),
        vocab,
        encoder,
        limit=train_config.eval_examples,
        batch_size=train_config.batch_size,
        synth=synth,
    )
    candidate_masks(eval_chords, index, eval_batches)
    print(f"  {sum(len(b) for b in eval_batches):,} validation examples", flush=True)

    decay, no_decay = [], []
    for name, param in model.head.named_parameters():
        (no_decay if "bias" in name else decay).append(param)
    optimiser = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": train_config.encoder_lr,
                "weight_decay": train_config.weight_decay,
            },
            {"params": decay, "lr": train_config.head_lr, "weight_decay": train_config.weight_decay},
            {"params": no_decay, "lr": train_config.head_lr, "weight_decay": 0.0},
        ]
    )
    base_lrs = [group["lr"] for group in optimiser.param_groups]
    loss_fn = nn.CrossEntropyLoss()

    stream = ExampleStream(
        vocab, encoder, corpus=corpus, split="train", synth=synth, seed=train_config.seed
    )
    history: list[dict] = []
    started = time.time()
    running = 0.0
    last_loss = float("nan")
    seen_examples = 0

    print(f"training {train_config.steps:,} steps x batch {train_config.batch_size}", flush=True)
    stepper = stream.prefetched_batches(train_config.batch_size, train_config.steps)
    for step, batch in enumerate(stepper):
        scale = learning_rate_scale(step, train_config)
        for group, base in zip(optimiser.param_groups, base_lrs):
            group["lr"] = base * scale

        moved = batch.to(device)
        use_amp = train_config.amp and device.type in {"mps", "cuda"}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(
                moved.input_ids, moved.attention_mask, moved.mask_position, moved.token_type_ids
            )
            loss = loss_fn(logits.float(), moved.target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

        running += float(loss.detach())
        seen_examples += len(batch)

        if (step + 1) % train_config.log_every == 0:
            elapsed = time.time() - started
            last_loss = running / train_config.log_every
            running = 0.0
            rate = seen_examples / elapsed
            print(
                f"  step {step + 1:>6,}/{train_config.steps:,}  loss {last_loss:6.3f}  "
                f"lr x{scale:4.2f}  {rate:7.1f} ex/s  {elapsed / 60:5.1f} min",
                flush=True,
            )

        if (step + 1) % train_config.eval_every == 0 or (step + 1) == train_config.steps:
            metrics = evaluate(model, eval_batches, device)
            metrics["step"] = step + 1
            metrics["loss"] = last_loss
            history.append(metrics)
            print(
                f"  eval @ {step + 1:,}  top1 {metrics['top1']:6.1%}  "
                f"top3 {metrics['top3']:6.1%}  top5 {metrics['top5']:6.1%}",
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": asdict(model_config),
                    "step": step + 1,
                    "metrics": metrics,
                },
                out_dir / "checkpoint.pt",
            )
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    total = time.time() - started
    summary = {
        "steps": train_config.steps,
        "examples": seen_examples,
        "minutes": total / 60,
        "examples_per_second": seen_examples / total,
        "history": history,
        "final": history[-1] if history else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"done: {seen_examples:,} examples in {total / 60:.1f} min "
        f"({seen_examples / total:.0f} ex/s)",
        flush=True,
    )
    return summary
