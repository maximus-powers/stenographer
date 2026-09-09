"""Messages between the tap process and the brain.

The two run separately for one reason. A process holding an active event tap freezes
keyboard input for every application on the machine if its callback stalls, and the
system's own recovery from that is unreliable. Torch holds the interpreter lock during
the Python side of a forward pass, and an Accessibility read can block for seconds on an
unresponsive app, so neither may sit in the same process as the tap.

That makes the boundary a real one, and this module is all of it: plain dataclasses over
newline-delimited JSON. Nothing here imports torch, numpy or pyobjc, so both sides can
load it without dragging in the other's dependencies.

The tap never waits indefinitely. If a prediction misses its deadline the tap commits the
most frequent candidate from its own index and drops the late reply, so stopping the brain
slows typing down but never blocks it.
"""

from __future__ import annotations

import json
# MISSING is the only way to tell a field with no default from one defaulting to None,
# which is what makes "this message is missing a field" distinguishable from "this field
# is None".
from dataclasses import MISSING as _MISSING
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROTOCOL_VERSION = 1

# How long the tap waits for a prediction before falling back to frequency alone.
DEFAULT_DEADLINE_MS = 50.0


# A Unix socket path is limited to about this many bytes on macOS, and exceeding it fails
# with a message that says nothing about which path was too long.
MAX_SOCKET_PATH = 100


def socket_path() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "stenographer"
        / "stenod.sock"
    )


def check_socket_path(path: Path) -> None:
    """Reject a socket path the operating system cannot use.

    Worth checking explicitly: binding a long path fails with "AF_UNIX path too long",
    which does not name the path, and temporary directories on macOS are long enough to
    run into this.
    """
    length = len(str(path).encode("utf-8"))
    if length > MAX_SOCKET_PATH:
        raise ValueError(
            f"socket path is {length} bytes, over the {MAX_SOCKET_PATH}-byte limit for a "
            f"Unix socket: {path}. Choose a shorter path, such as one under /tmp."
        )


class Message:
    """Base class for everything crossing the socket."""

    @property
    def kind(self) -> str:
        return type(self).__name__


# -- tap to brain ---------------------------------------------------------------


@dataclass(frozen=True)
class PredictRequest(Message):
    """A chord was detected; name the word it meant."""

    chord_id: int
    letters: str  # sorted distinct chord letters: the chord's identity
    press_order: str = ""  # order the keys went down, a weak signal at best
    shift: bool = False
    caps_lock: bool = False
    app_bundle_id: str = ""
    t_ms: float = 0.0


@dataclass(frozen=True)
class Typed(Message):
    """Characters that reached the application, so the brain can follow the text."""

    text: str


@dataclass(frozen=True)
class Nav(Message):
    """Something happened that may have moved the caret.

    The brain keeps its own copy of the text before the caret, and that copy is only
    trustworthy while the caret moves in the one way it can predict: forwards, one typed
    character at a time. Anything else invalidates it.
    """

    kind_name: str  # click, arrow, enter, tab, backspace, app_switch, focus_change


@dataclass(frozen=True)
class CommitResult(Message):
    """What the tap actually typed, including when it fell back to its own index."""

    chord_id: int
    inserted_text: str
    source: str = "lm"  # lm, fallback, or letters


# -- brain to tap ---------------------------------------------------------------


@dataclass(frozen=True)
class Prediction(Message):
    """The word to type, already cased, plus the alternatives the cycle key steps through."""

    chord_id: int
    insert_text: str
    candidates: tuple[str, ...] = ()
    source: str = "lm"
    needs_leading_space: bool = False


@dataclass(frozen=True)
class ModeUpdate(Message):
    """Pushed when the focused application or field changes.

    Pushed rather than polled: asking the brain about the focused field on every
    keystroke would put an Accessibility call on the critical path, which is exactly the
    blocking work the process split exists to keep away from the tap.
    """

    enabled: bool = True
    app_rule: str = "allow"  # allow, deny
    secure_input: bool = False
    code_context: bool = False


MESSAGES: dict[str, type] = {
    cls.__name__: cls
    for cls in (PredictRequest, Typed, Nav, CommitResult, Prediction, ModeUpdate)
}


# -- wire format ----------------------------------------------------------------


def encode(message: Message) -> bytes:
    """Serialise one message as a single line of JSON."""
    payload = {"v": PROTOCOL_VERSION, "kind": message.kind, "body": asdict(message)}
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes | str) -> Message:
    """Parse one line back into a message.

    Raises ValueError on anything unrecognised. The socket carries only our own traffic,
    but a version mismatch between a running daemon and a freshly started tap is an
    ordinary thing to hit during development and should say so plainly.
    """
    text = line.decode("utf-8") if isinstance(line, bytes) else line
    text = text.strip()
    if not text:
        raise ValueError("empty message")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("message must be a JSON object")

    version = payload.get("v")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"protocol version {version!r}, expected {PROTOCOL_VERSION}")

    kind = payload.get("kind")
    message_type = MESSAGES.get(str(kind))
    if message_type is None:
        raise ValueError(f"unknown message kind {kind!r}")

    body = payload.get("body")
    if not isinstance(body, dict):
        raise ValueError("message body must be a JSON object")

    known = {f.name for f in fields(message_type)}
    unknown = set(body) - known
    if unknown:
        raise ValueError(f"{kind} has unexpected fields {sorted(unknown)}")
    missing = {
        f.name
        for f in fields(message_type)
        if f.name not in body and f.default is f.default_factory is _MISSING
    }
    if missing:
        raise ValueError(f"{kind} is missing fields {sorted(missing)}")

    prepared = dict(body)
    for field_info in fields(message_type):
        if field_info.type == "tuple[str, ...]" and field_info.name in prepared:
            prepared[field_info.name] = tuple(prepared[field_info.name])
    return message_type(**prepared)


@dataclass
class LineReader:
    """Accumulates socket bytes and yields whole messages.

    A stream gives no guarantee that a read lands on a message boundary, so partial lines
    are held until their newline arrives.
    """

    buffer: bytearray = field(default_factory=bytearray)

    def feed(self, chunk: bytes) -> list[Message]:
        self.buffer.extend(chunk)
        messages: list[Message] = []
        while True:
            index = self.buffer.find(b"\n")
            if index < 0:
                return messages
            line = bytes(self.buffer[:index])
            del self.buffer[: index + 1]
            if line.strip():
                messages.append(decode(line))
