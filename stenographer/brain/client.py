"""The tap's connection to the prediction daemon.

Every call here has a deadline, and missing it is not an error. If the daemon is slow,
busy, crashed, or simply not running, the tap falls back to the most frequent word
containing the chord letters, which it can find in microseconds from its own index. The
prediction is an improvement on that answer, never a prerequisite for having one.

That is the whole reason the deadline exists. This code runs inside the event tap
callback, and a callback that waits freezes the keyboard for every application on the
machine.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

from stenographer.protocol import (
    DEFAULT_DEADLINE_MS,
    LineReader,
    Message,
    Prediction,
    check_socket_path,
    encode,
    socket_path,
)


@dataclass
class ClientStats:
    requests: int = 0
    answered: int = 0
    timeouts: int = 0
    disconnects: int = 0
    total_ms: float = 0.0
    slowest_ms: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.answered if self.answered else 0.0


@dataclass
class BrainClient:
    """A best-effort request-reply channel to the daemon."""

    path: Path = field(default_factory=socket_path)
    deadline_ms: float = DEFAULT_DEADLINE_MS
    stats: ClientStats = field(default_factory=ClientStats)
    connection: socket.socket | None = None
    reader: LineReader = field(default_factory=LineReader)
    unusable: str = ""  # set when the path itself can never work

    # -- connection -----------------------------------------------------------

    def connect(self) -> bool:
        """Try to connect. Returns whether a connection is available."""
        if self.connection is not None:
            return True
        if self.unusable:
            return False
        try:
            check_socket_path(self.path)
        except ValueError as error:
            # A permanent problem, so record it and stop retrying on every chord.
            self.unusable = str(error)
            return False
        if not self.path.exists():
            return False
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.deadline_ms / 1000.0)
            sock.connect(str(self.path))
        except OSError:
            return False
        self.connection = sock
        self.reader = LineReader()
        return True

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None

    def _drop(self) -> None:
        self.stats.disconnects += 1
        self.close()

    # -- messaging ------------------------------------------------------------

    def notify(self, message: Message) -> None:
        """Send something the daemon does not answer, such as text that was typed."""
        if not self.connect():
            return
        assert self.connection is not None
        try:
            self.connection.sendall(encode(message))
        except OSError:
            self._drop()

    def request(self, message: Message) -> Prediction | None:
        """Send a request and wait up to the deadline for its answer.

        Returns None if the daemon is unavailable or too slow, which tells the caller to
        use its own fallback.
        """
        self.stats.requests += 1
        if not self.connect():
            return None
        assert self.connection is not None

        started = time.perf_counter()
        try:
            self.connection.sendall(encode(message))
        except OSError:
            self._drop()
            return None

        remaining = self.deadline_ms / 1000.0
        while remaining > 0:
            try:
                self.connection.settimeout(remaining)
                chunk = self.connection.recv(4096)
            except (socket.timeout, TimeoutError):
                break
            except OSError:
                self._drop()
                return None
            if not chunk:
                self._drop()
                return None

            for reply in self.reader.feed(chunk):
                if isinstance(reply, Prediction):
                    elapsed = (time.perf_counter() - started) * 1000.0
                    self.stats.answered += 1
                    self.stats.total_ms += elapsed
                    self.stats.slowest_ms = max(self.stats.slowest_ms, elapsed)
                    # A reply to a request we already gave up on: ignore it and keep
                    # waiting for the one we asked about.
                    if getattr(message, "chord_id", None) == reply.chord_id:
                        return reply
            remaining = self.deadline_ms / 1000.0 - (time.perf_counter() - started)

        self.stats.timeouts += 1
        return None
