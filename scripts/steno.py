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
import signal
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
from stenographer.keyboard.layout import BACKSPACE, RIGHT_OPTION, TAB
from stenographer.keyboard.tap import (
    EventTap,
    accessibility_trusted,
    secure_input_enabled,
)
from stenographer.protocol import Nav, PredictRequest, Typed, socket_path
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
            chord_max_ms=args.chord_max,
            lone_hold_ms=args.lone_hold,
            two_key_overlap_ms=args.overlap,
            min_overlap_ms=args.min_overlap,
            min_keys=args.min_keys,
            max_keys=args.max_keys,
            trigger=Trigger(args.trigger),
        )
    )
    composer = TextComposer(ComposerConfig(auto_space=not args.no_auto_space))

    def on_text(text: str, backspaces: int = 0) -> None:
        if client is not None:
            client.notify(Typed(text=text, backspaces=backspaces))

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
        # Only hold-to-chord reads this key. Passing it in the other modes would exempt
        # right Option from being a shortcut for no reason.
        chord_key=args.chord_key if args.trigger == "hold_to_chord" else None,
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
    # Defaults come from DetectorConfig rather than being written out again. They were
    # written out, and they drifted: --overlap still said 60 ms after the dataclass moved
    # to 150, and since the flag is always passed the dataclass value never reached the
    # detector at all.
    timing = DetectorConfig()
    parser.add_argument(
        "--chord-max",
        type=float,
        default=timing.chord_max_ms,
        help="ms a whole chord gesture may take",
    )
    parser.add_argument(
        "--lone-hold",
        type=float,
        default=timing.lone_hold_ms,
        help="ms to hold a single key back",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=timing.two_key_overlap_ms,
        help="ms a pair must be held together",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=timing.min_overlap_ms,
        help="ms every key in a group must be held together",
    )
    parser.add_argument("--min-keys", type=int, default=timing.min_keys)
    parser.add_argument("--max-keys", type=int, default=timing.max_keys)
    # Always on, which is the point: nothing to hold, nothing to remember. What makes it
    # safe is the duration rule in the detector rather than this choice - before that, a
    # group of three or more became a chord on a single instant with every key down, and
    # ordinary rollover produces those constantly. `--trigger hold_to_chord` is the
    # fallback if the thresholds still do not suit a particular pair of hands, and it
    # cannot false-fire at all.
    parser.add_argument(
        "--trigger", default="holdback", choices=[t.value for t in Trigger]
    )
    parser.add_argument("--cycle-key", default="tab", choices=["tab", "right-option"])
    parser.add_argument(
        "--chord-key",
        default="right-option",
        choices=["right-option", "fn"],
        help="key to hold for a chord, with --trigger hold_to_chord",
    )
    parser.add_argument("--no-auto-space", action="store_true")
    parser.add_argument("--gap", type=float, default=0.0, help="ms between injected events")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Line-buffer output: this process spends its life inside a run loop, and a banner or
    # a dry-run decision held in a pipe buffer is of no use to anyone watching.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    if not args.dry_run and not accessibility_trusted():
        print("Accessibility permission is not granted, so nothing would be captured.")
        print("Run: python scripts/check_permissions.py")
        raise SystemExit(1)

    if args.trigger == "hold_to_chord" and args.chord_key == args.cycle_key:
        print(
            f"--chord-key and --cycle-key are both {args.cycle_key}. "
            "Holding it would open a chord and cycle the last one at the same time."
        )
        raise SystemExit(2)

    runtime, tap, emitter, predictor, client = build(args)

    print("stenographer")
    print(f"  mode          {'dry run, nothing is typed' if args.dry_run else 'live'}")
    if args.trigger == "hold_to_chord":
        held = "right Option" if args.chord_key == "right-option" else "Fn"
        print(f"  trigger       hold {held} and press the word's first letters")
    else:
        print(f"  trigger       {args.trigger}, {args.min_keys}+ keys")
    print(
        f"  timing        up to {args.chord_max:.0f} ms per chord, held together "
        f"{args.min_overlap:.0f} ms ({args.overlap:.0f} for a pair), "
        f"single key held {args.lone_hold:.0f} ms"
    )
    print(f"  predictions   {'local frequency index' if client is None else f'daemon at {args.socket or socket_path()}'}")
    if secure_input_enabled():
        print("  note          Secure Input is on; no keystrokes will be seen")
    print("  Ctrl-C to stop\n")

    # Ask the run loop to finish rather than relying on the default signal behaviour.
    # Ctrl-C raising KeyboardInterrupt is not something to depend on here: this process
    # spends its life inside a run loop, and a background process started by a
    # non-interactive shell inherits SIGINT ignored altogether. Since this one sits
    # between the user and their keyboard, stopping it has to work every time.
    def shutdown(signum, _frame) -> None:
        print(f"\n  stopping on {signal.Signals(signum).name}")
        tap.running = False

    for received in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(received, shutdown)

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
