"""Protocol tests.

The socket is the boundary between the two processes, so a message that fails to round
trip means the tap and the brain disagree about what was typed. These tests also pin the
failure behaviour: a malformed or mismatched message must raise something readable rather
than silently producing a half-populated object.
"""

from __future__ import annotations

import pytest

from stenographer.protocol import (
    PROTOCOL_VERSION,
    CommitResult,
    LineReader,
    ModeUpdate,
    Nav,
    PredictRequest,
    Prediction,
    Typed,
    decode,
    encode,
)

EXAMPLES = [
    PredictRequest(chord_id=1, letters="eht", press_order="the", app_bundle_id="com.apple.TextEdit"),
    PredictRequest(chord_id=2, letters="ot", shift=True, caps_lock=True, t_ms=1234.5),
    Typed(text="hello "),
    Nav(kind_name="click"),
    CommitResult(chord_id=1, inserted_text="there ", source="fallback"),
    Prediction(chord_id=1, insert_text="There ", candidates=("there", "their"), source="lm"),
    Prediction(chord_id=3, insert_text="the ", needs_leading_space=True),
    ModeUpdate(enabled=False, app_rule="deny", secure_input=True, code_context=True),
]


@pytest.mark.parametrize("message", EXAMPLES, ids=lambda m: f"{m.kind}")
def test_round_trip(message):
    assert decode(encode(message)) == message


@pytest.mark.parametrize("message", EXAMPLES, ids=lambda m: f"{m.kind}")
def test_encoding_is_one_line(message):
    wire = encode(message)
    assert wire.endswith(b"\n")
    assert wire.count(b"\n") == 1


def test_candidates_come_back_as_a_tuple():
    """JSON has only arrays, so the tuple has to be restored explicitly."""
    restored = decode(encode(Prediction(chord_id=1, insert_text="x ", candidates=("a", "b"))))
    assert isinstance(restored.candidates, tuple)
    assert restored.candidates == ("a", "b")


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown message kind"):
        decode('{"v": %d, "kind": "Nonsense", "body": {}}' % PROTOCOL_VERSION)


def test_version_mismatch_is_rejected_with_the_version_named():
    with pytest.raises(ValueError, match="protocol version"):
        decode('{"v": 999, "kind": "Typed", "body": {"text": "x"}}')


def test_unexpected_field_is_rejected():
    with pytest.raises(ValueError, match="unexpected fields"):
        decode('{"v": %d, "kind": "Typed", "body": {"text": "x", "extra": 1}}' % PROTOCOL_VERSION)


def test_missing_required_field_is_rejected():
    with pytest.raises(ValueError, match="missing fields"):
        decode('{"v": %d, "kind": "Typed", "body": {}}' % PROTOCOL_VERSION)


@pytest.mark.parametrize("bad", ["", "   ", "not json", "[1,2,3]", '{"v": 1}'])
def test_malformed_input_raises_value_error(bad):
    with pytest.raises(ValueError):
        decode(bad)


# -- streaming ------------------------------------------------------------------


def test_reader_yields_complete_messages():
    reader = LineReader()
    wire = b"".join(encode(m) for m in EXAMPLES)
    assert reader.feed(wire) == EXAMPLES


def test_reader_holds_a_partial_message_until_it_is_complete():
    reader = LineReader()
    wire = encode(Typed(text="hello"))
    assert reader.feed(wire[:6]) == []
    assert reader.feed(wire[6:]) == [Typed(text="hello")]


def test_reader_handles_a_message_split_across_many_reads():
    reader = LineReader()
    wire = encode(PredictRequest(chord_id=7, letters="abc"))
    received = []
    for index in range(len(wire)):
        received.extend(reader.feed(wire[index : index + 1]))
    assert received == [PredictRequest(chord_id=7, letters="abc")]


def test_reader_ignores_blank_lines():
    reader = LineReader()
    assert reader.feed(b"\n\n" + encode(Typed(text="x"))) == [Typed(text="x")]


def test_reader_keeps_leftovers_between_calls():
    reader = LineReader()
    first, second = encode(Typed(text="one")), encode(Typed(text="two"))
    combined = first + second
    cut = len(first) + 4
    assert reader.feed(combined[:cut]) == [Typed(text="one")]
    assert reader.feed(combined[cut:]) == [Typed(text="two")]
