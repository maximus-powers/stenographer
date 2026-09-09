"""Brain tests: context tracking, casing, and the fallback path.

The context the brain keeps is the model's only view of what the user is writing, and it
is built entirely from what the tap reports. Two things must hold: it follows what
actually reached the application rather than what was suggested, and it gives up cleanly
whenever the caret may have moved somewhere it cannot see.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from stenographer.brain.client import BrainClient
from stenographer.brain.server import CONTEXT_CHARS, Brain, Context
from stenographer.protocol import (
    CommitResult,
    ModeUpdate,
    Nav,
    Prediction,
    PredictRequest,
    Typed,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def short_tmp_path(tmp_path):
    """A directory short enough to hold a Unix socket.

    pytest's own temporary paths run past the operating system's limit of about 100 bytes
    for a socket path, so tests that bind one need somewhere shorter.
    """
    import shutil
    import tempfile

    directory = Path(tempfile.mkdtemp(prefix="steno-", dir="/tmp"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


class StubPredictor:
    """Returns fixed candidates and records the context it was given."""

    def __init__(self, candidates=("there", "their", "they")) -> None:
        self.candidates = list(candidates)
        self.seen: list = []

    def predict(self, request, k=5):
        self.seen.append(request)
        return self.candidates[:k]


# -- context --------------------------------------------------------------------


def test_context_accumulates_typed_text():
    context = Context()
    context.add("I went ")
    context.add("there ")
    assert context.text == "I went there "
    assert context.words() == ["I", "went", "there"]


def test_context_is_bounded():
    context = Context()
    context.add("x" * (CONTEXT_CHARS * 3))
    assert len(context.text) == CONTEXT_CHARS


def test_context_keeps_the_most_recent_text():
    context = Context()
    context.add("old " * 200)
    context.add("the newest words")
    assert context.text.endswith("the newest words")


def test_invalidating_context_marks_it_unknown():
    context = Context()
    context.add("I went ")
    context.invalidate()
    assert context.text == ""
    assert context.known is False


def test_typing_again_makes_context_known():
    context = Context()
    context.invalidate()
    context.add("new ")
    assert context.known is True


# -- message handling -----------------------------------------------------------


def test_typed_text_reaches_the_predictor_as_context():
    predictor = StubPredictor()
    brain = Brain(predictor=predictor)
    brain.handle(Typed(text="I went "))
    brain.handle(PredictRequest(chord_id=1, letters="eht"))
    assert predictor.seen[-1].left == ("I", "went")


def test_navigation_clears_the_context():
    predictor = StubPredictor()
    brain = Brain(predictor=predictor)
    brain.handle(Typed(text="I went "))
    brain.handle(Nav(kind_name="click"))
    brain.handle(PredictRequest(chord_id=1, letters="eht"))
    assert predictor.seen[-1].left == ()


def test_context_follows_what_was_actually_typed_not_what_was_suggested():
    """The tap may have used its own fallback word, so that is what the context records."""
    predictor = StubPredictor()
    brain = Brain(predictor=predictor)
    brain.handle(CommitResult(chord_id=1, inserted_text="them ", source="fallback"))
    brain.handle(PredictRequest(chord_id=2, letters="eht"))
    assert predictor.seen[-1].left == ("them",)


def test_a_prediction_is_returned_with_its_chord_id():
    brain = Brain(predictor=StubPredictor())
    reply = brain.handle(PredictRequest(chord_id=42, letters="eht"))
    assert isinstance(reply, Prediction)
    assert reply.chord_id == 42
    assert reply.candidates == ("there", "their", "they")


def test_no_candidates_gives_an_empty_prediction():
    brain = Brain(predictor=StubPredictor(candidates=()))
    reply = brain.handle(PredictRequest(chord_id=1, letters="xqj"))
    assert reply.insert_text == ""
    assert reply.source == "none"


def test_notifications_get_no_reply():
    brain = Brain(predictor=StubPredictor())
    assert brain.handle(Typed(text="x")) is None
    assert brain.handle(Nav(kind_name="click")) is None


# -- casing ---------------------------------------------------------------------


def test_the_first_candidate_is_cased_for_its_position():
    brain = Brain(predictor=StubPredictor())
    brain.handle(Typed(text="I stopped. "))
    reply = brain.handle(PredictRequest(chord_id=1, letters="eht"))
    assert reply.insert_text == "There"
    # The alternatives stay as the vocabulary has them; the tap cases them when used.
    assert reply.candidates[0] == "there"


def test_mid_sentence_stays_lowercase():
    brain = Brain(predictor=StubPredictor())
    brain.handle(Typed(text="I went "))
    reply = brain.handle(PredictRequest(chord_id=1, letters="eht"))
    assert reply.insert_text == "there"


def test_shift_capitalises():
    brain = Brain(predictor=StubPredictor())
    brain.handle(Typed(text="I went "))
    reply = brain.handle(PredictRequest(chord_id=1, letters="eht", shift=True))
    assert reply.insert_text == "There"


def test_caps_lock_uppercases():
    brain = Brain(predictor=StubPredictor())
    brain.handle(Typed(text="I went "))
    reply = brain.handle(PredictRequest(chord_id=1, letters="eht", caps_lock=True))
    assert reply.insert_text == "THERE"


def test_unknown_context_does_not_capitalise():
    brain = Brain(predictor=StubPredictor())
    brain.handle(Nav(kind_name="click"))
    reply = brain.handle(PredictRequest(chord_id=1, letters="eht"))
    assert reply.insert_text == "there"


def test_code_context_suppresses_capitalisation():
    brain = Brain(predictor=StubPredictor(), mode=ModeUpdate(code_context=True))
    brain.handle(Typed(text="value = compute(). "))
    reply = brain.handle(PredictRequest(chord_id=1, letters="eht"))
    assert reply.insert_text == "there"


# -- statistics -----------------------------------------------------------------


def test_slow_predictions_are_counted_against_the_deadline():
    class Slow(StubPredictor):
        def predict(self, request, k=5):
            time.sleep(0.02)
            return super().predict(request, k)

    brain = Brain(predictor=Slow(), deadline_ms=5.0)
    brain.handle(PredictRequest(chord_id=1, letters="eht"))
    assert brain.stats.requests == 1
    assert brain.stats.over_deadline == 1
    assert brain.stats.slowest_ms >= 20.0


# -- the client's failure behaviour ---------------------------------------------


def test_client_returns_nothing_promptly_when_no_daemon_is_running(short_tmp_path):
    """A missing daemon must not cost the deadline; the tap has a word to fall back to."""
    client = BrainClient(path=short_tmp_path / "absent.sock", deadline_ms=500.0)
    started = time.perf_counter()
    assert client.request(PredictRequest(chord_id=1, letters="eht")) is None
    assert (time.perf_counter() - started) * 1000 < 100.0


def test_client_notify_is_harmless_without_a_daemon(short_tmp_path):
    client = BrainClient(path=short_tmp_path / "absent.sock")
    client.notify(Typed(text="x"))  # must not raise


def test_client_gives_up_on_a_daemon_that_never_answers(short_tmp_path):
    """A socket that accepts and then says nothing must still hit the deadline, once."""
    path = short_tmp_path / "silent.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    try:
        client = BrainClient(path=path, deadline_ms=30.0)
        started = time.perf_counter()
        assert client.request(PredictRequest(chord_id=1, letters="eht")) is None
        elapsed = (time.perf_counter() - started) * 1000
        assert 20.0 <= elapsed < 300.0
        assert client.stats.timeouts == 1
    finally:
        server.close()


@pytest.mark.slow
def test_the_two_processes_talk_to_each_other(short_tmp_path):
    """Start the real daemon and ask it a real question."""
    path = short_tmp_path / "stenod.sock"
    if not (ROOT / "data" / "vocab" / "vocab.json").exists():
        pytest.skip("no vocabulary built; run scripts/build_vocab.py")

    process = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "stenod.py"), "--no-model", "--socket", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=ROOT,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not path.exists():
            time.sleep(0.1)
        assert path.exists(), "daemon did not create its socket"

        client = BrainClient(path=path, deadline_ms=2000.0)
        client.notify(Typed(text="I went "))
        reply = client.request(PredictRequest(chord_id=1, letters="eht"))
        assert reply is not None
        assert reply.insert_text == "the"
        client.close()
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_an_overlong_socket_path_is_reported_clearly(tmp_path):
    """pytest's own temporary paths are long enough to trigger this."""
    client = BrainClient(path=tmp_path / ("x" * 120) / "stenod.sock")
    assert client.connect() is False
    assert "over the" in client.unusable and "Unix socket" in client.unusable
