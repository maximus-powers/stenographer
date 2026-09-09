"""The prediction daemon.

Runs the model and answers the tap over a Unix socket. It lives in its own process for
one reason, given at length in `stenographer/protocol.py`: a stalled event-tap callback
freezes keyboard input for every application on the machine, and torch holds the
interpreter lock through the Python side of a forward pass. Keeping the model out here
means the worst a slow or crashed model can do is make chording fall back to frequency
alone.

It also keeps its own copy of the text before the caret, built from what the tap reports
having typed. That is the context the model conditions on, and it is deliberately
maintained here rather than read from the focused application on every chord, because an
Accessibility read can block for seconds and would put that delay on the critical path.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from stenographer.brain.predictor.base import Request
from stenographer.protocol import (
    CommitResult,
    LineReader,
    ModeUpdate,
    Nav,
    Prediction,
    PredictRequest,
    Typed,
    check_socket_path,
    encode,
    socket_path,
)
from stenographer.text.casing import CasingContext
from stenographer.text.casing import apply as apply_casing

# How much text before the caret to keep. The model reads at most about 48 wordpieces,
# so a few hundred characters is more than enough.
CONTEXT_CHARS = 400
CANDIDATES = 5


@dataclass
class Context:
    """What has been typed, as far as the tap has reported it."""

    text: str = ""
    known: bool = True

    def add(self, addition: str) -> None:
        self.text = (self.text + addition)[-CONTEXT_CHARS:]
        self.known = True

    def invalidate(self) -> None:
        """The caret may have moved somewhere we cannot follow."""
        self.text = ""
        self.known = False

    def words(self, count: int = 48) -> list[str]:
        return self.text.split()[-count:]


@dataclass
class ServerStats:
    requests: int = 0
    total_ms: float = 0.0
    slowest_ms: float = 0.0
    over_deadline: int = 0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.requests if self.requests else 0.0


@dataclass
class Brain:
    """Answers prediction requests and tracks the text around the caret."""

    predictor: object  # anything with predict(Request, k) -> list[str]
    context: Context = field(default_factory=Context)
    mode: ModeUpdate = field(default_factory=ModeUpdate)
    stats: ServerStats = field(default_factory=ServerStats)
    deadline_ms: float = 50.0

    def handle(self, message) -> Prediction | None:
        if isinstance(message, PredictRequest):
            return self._predict(message)
        if isinstance(message, Typed):
            self.context.add(message.text)
            return None
        if isinstance(message, Nav):
            self.context.invalidate()
            return None
        if isinstance(message, CommitResult):
            # The tap may have typed the fallback word instead of ours, so the context
            # follows what actually reached the application, not what we suggested.
            self.context.add(message.inserted_text)
            return None
        return None

    def _predict(self, request: PredictRequest) -> Prediction:
        started = time.perf_counter()
        candidates = self.predictor.predict(
            Request(
                letters=request.letters,
                press_order=request.press_order,
                left=tuple(self.context.words()),
                sentence_start=False,
                shift=request.shift,
                caps_lock=request.caps_lock,
            ),
            CANDIDATES,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        self.stats.requests += 1
        self.stats.total_ms += elapsed
        self.stats.slowest_ms = max(self.stats.slowest_ms, elapsed)
        if elapsed > self.deadline_ms:
            self.stats.over_deadline += 1

        if not candidates:
            return Prediction(
                chord_id=request.chord_id, insert_text="", candidates=(), source="none"
            )

        context = CasingContext(
            preceding=self.context.text,
            shift=request.shift,
            caps_lock=request.caps_lock,
            known=self.context.known,
        )
        word = apply_casing(candidates[0], context, code_context=self.mode.code_context)
        return Prediction(
            chord_id=request.chord_id,
            insert_text=word,
            candidates=tuple(candidates),
            source="lm",
        )


class BrainServer:
    """Serves one or more tap processes over a Unix socket."""

    def __init__(self, brain: Brain, path: Path | None = None, verbose: bool = False) -> None:
        self.brain = brain
        self.path = path or socket_path()
        self.verbose = verbose
        self.server: asyncio.AbstractServer | None = None

    async def serve(self) -> None:
        check_socket_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            # A socket left behind by a process that did not shut down cleanly.
            self.path.unlink()
        self.server = await asyncio.start_unix_server(self._client, path=str(self.path))
        print(f"listening on {self.path}", flush=True)
        async with self.server:
            await self.server.serve_forever()

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.verbose:
            print("tap connected", flush=True)
        lines = LineReader()
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                for message in lines.feed(chunk):
                    try:
                        reply = self.brain.handle(message)
                    except Exception as error:  # a bad prediction must not kill the server
                        print(f"prediction failed: {error!r}", flush=True)
                        continue
                    if reply is not None:
                        writer.write(encode(reply))
                        await writer.drain()
                        if self.verbose:
                            print(
                                f"  {message.letters!r} -> {reply.insert_text!r} "
                                f"{list(reply.candidates[:4])}",
                                flush=True,
                            )
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            if self.verbose:
                print("tap disconnected", flush=True)
