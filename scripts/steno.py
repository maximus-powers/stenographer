"""Run the keystroke side: watch for chords and type the words they mean.

Start with --dry-run. It watches without touching anything, printing what it would have
done, and it is the only safe way to find out whether the timing thresholds suit your
hands before letting it rewrite what you type.

    python scripts/steno.py --dry-run          # watch and report, change nothing
    python scripts/steno.py                    # for real

Needs Accessibility permission, which fails silently when missing. Run
`scripts/check_permissions.py` first.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.candidates import FallbackIndex
from model.vocab import Vocab
from stenographer.brain.client import BrainClient
from stenographer.keyboard.composer import ComposerConfig, TextComposer
from stenographer.keyboard.detector import (
    ChordDetector,
    CommitChord,
    DetectorConfig,
    Trigger,
)
from stenographer.keyboard.inject import Emitter
from stenographer.keyboard.layout import RIGHT_OPTION, TAB, BACKSPACE
from stenographer.keyboard.tap import (
    EventTap,
    accessibility_trusted,
    secure_input_enabled,
)
from stenographer.protocol import CommitResult, Nav, PredictRequest, Typed, socket_path
from stenographer.runtime import Runtime


class Predictor:
    """Asks the daemon, and answers from the local index when it cannot.

    The local index is the reason chording keeps working when the daemon is stopped,
    slow, or was never started. It is a bitwise intersection over the vocabulary, so it
    costs microseconds and is safe to call from inside the tap callback.
    """

    def __init__(self, vocab: Vocab, client: BrainClient | None, verbose: bool = False) -> None:
        self.fallback = FallbackIndex.from_vocab(vocab)
        self.client = client
        self.verbose = verbose
        self.next_id = 0
        self.sources: Counter[str] = Counter()
        self.last_source = "fallback"

    def __call__(self, chord: CommitChord) -> list[str]:
        self.next_id += 1
        if self.client is not None:
            reply = self.client.request(
                PredictRequest(
                    chord_id=self.next_id,
                    letters=chord.letters,
                    press_order=chord.press_order,
                    shift=chord.shift,
                    caps_lock=chord.caps_lock,
                )
            )
            if reply is not None and reply.candidates:
                self.sources["model"] += 1
                self.last_source = "model"
                # The daemon has already applied casing to the first candidate.
                return [reply.insert_text, *reply.candidates[1:]]

        words = self.fallback.top_k(chord.letters, k=5)
        self.sources["fallback" if words else "none"] += 1
        self.last_source = "fallback" if words else "none"
        return words


def build(args) -> tuple[Runtime, EventTap, Emitter, Predictor, BrainClient | None]:
    vocab = Vocab.load(args.vocab)
    client = None if args.no_daemon else BrainClient(
        path=args.socket or socket_path(), deadline_ms=args.deadline
    )
    predictor = Predictor(vocab, client, verbose=args.verbose)
    emitter = Emitter(dry_run=args.dry_run, gap_seconds=args.gap / 1000.0)

    detector = ChordDetector(
        DetectorConfig(
            join_ms=args.join,
            late_ms=args.late,
            two_key_overlap_ms=args.overlap,
            min_keys=args.min_keys,
            max_keys=args.max_keys,
            trigger=Trigger(args.trigger),
        )
    )
    composer = TextComposer(ComposerConfig(auto_space=not args.no_auto_space))

    typed_total = Counter()

    def on_text(text: str) -> None:
        typed_total["characters"] += len(text)
        if client is not None:
            client.notify(Typed(text=text))

    def on_nav(_kind: str) -> None:
        if client is not None:
            client.notify(Nav(kind_name="navigate"))

    runtime = Runtime(
        detector=detector,
        composer=composer,
        emitter=emitter,
        predict=predictor,
        cycle_keycode=TAB if args.cycle_key == "tab" else RIGHT_OPTION,
        undo_keycode=BACKSPACE,
        on_text=on_text,
        on_nav=on_nav,
    )

    log: list[str] | None = [] if args.dry_run else None
    if log is not None:
        emitter.log = log

    def handler(key_event, raw) -> bool:
        before = len(log) if log is not None else 0
        keep = runtime.handle(key_event, raw)
        if log is not None and len(log) > before:
            for line in log[before:]:
                marker = "would" if args.dry_run else "did"
                print(f"    {marker} {line}  [{predictor.last_source}]", flush=True)
        return keep

    tap = EventTap(
        handler,
        listen_only=args.dry_run,
        on_timer=runtime.on_timer,
        on_lost_events=lambda: runtime.detector.reset(),
    )
    runtime.layout = tap.layout
    return runtime, tap, emitter, predictor, client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="watch only, change nothing")
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab"))
    parser.add_argument("--socket", type=Path, default=None)
    parser.add_argument("--no-daemon", action="store_true", help="frequency index only")
    parser.add_argument("--deadline", type=float, default=50.0, help="ms to wait for the daemon")
    parser.add_argument("--join", type=float, default=50.0, help="ms for keys to join a chord")
    parser.add_argument("--late", type=float, default=120.0, help="ms for a late finger")
    parser.add_argument("--overlap", type=float, default=60.0, help="ms of overlap a pair needs")
    parser.add_argument("--min-keys", type=int, default=2)
    parser.add_argument("--max-keys", type=int, default=6)
    parser.add_argument(
        "--trigger", default="holdback", choices=[t.value for t in Trigger]
    )
    parser.add_argument("--cycle-key", default="tab", choices=["tab", "right-option"])
    parser.add_argument("--no-auto-space", action="store_true")
    parser.add_argument("--gap", type=float, default=0.0, help="ms between injected events")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not accessibility_trusted():
        print("Accessibility permission is not granted, so nothing would be captured.")
        print("Run: python scripts/check_permissions.py")
        raise SystemExit(1)

    runtime, tap, emitter, predictor, client = build(args)

    print("stenographer")
    print(f"  mode          {'dry run, nothing is typed' if args.dry_run else 'live'}")
    print(f"  trigger       {args.trigger}, {args.min_keys}+ keys")
    print(f"  timing        join {args.join:.0f} ms, late {args.late:.0f} ms, overlap {args.overlap:.0f} ms")
    print(f"  predictions   {'local frequency index' if client is None else f'daemon at {args.socket or socket_path()}'}")
    if secure_input_enabled():
        print("  note          Secure Input is on; no keystrokes will be seen")
    print("  Ctrl-C to stop\n")

    started = time.monotonic()
    try:
        tap.run()
    except PermissionError as error:
        print(f"\n{error}")
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        pass
    finally:
        tap.stop()
        if client is not None:
            client.close()
        elapsed = time.monotonic() - started
        stats = runtime.stats
        print(f"\n  ran for {elapsed / 60:.1f} min")
        print(f"  chords committed      {stats.chords}")
        print(f"  letters typed through {stats.letters}")
        print(f"  cycles / undos        {stats.cycles} / {stats.undos}")
        print(f"  chords with no match  {stats.no_candidates}")
        if predictor.sources:
            print(f"  answered by           {dict(predictor.sources)}")
        if client is not None and client.stats.requests:
            print(
                f"  daemon                {client.stats.answered}/{client.stats.requests} "
                f"answered, mean {client.stats.mean_ms:.1f} ms, "
                f"{client.stats.timeouts} timed out"
            )
        print(
            f"  tap callback          {tap.stats.events:,} events, "
            f"mean {tap.stats.mean_callback_ms:.3f} ms, peak {tap.stats.max_callback_ms:.1f} ms"
        )
        if tap.stats.reenabled:
            print(f"  tap re-enabled        {tap.stats.reenabled} times after being cut off")


if __name__ == "__main__":
    main()
