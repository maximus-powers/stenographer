import random

from model import corpus
from model.corpus import Example, count_words, iter_examples, tokenise


def test_tokenise_keeps_apostrophes_and_splits_hyphens():
    assert tokenise("Don't go late-night") == ["Don't", "go", "late", "-", "night"]


def test_tokenise_separates_punctuation():
    assert tokenise("Hi, there!") == ["Hi", ",", "there", "!"]


def test_examples_carry_left_and_right_context():
    examples = list(iter_examples(["The cat sat on the mat"]))
    targets = [e.target for e in examples]
    assert targets == ["The", "cat", "sat", "on", "the", "mat"]
    assert examples[2].left == ["The", "cat"]
    assert examples[2].right == ["on", "the", "mat"]


def test_sentence_start_is_detected():
    examples = {e.target: e.sentence_start for e in iter_examples(["It rained. Then it stopped."])}
    assert examples["It"] is True
    assert examples["Then"] is True
    assert examples["rained"] is False


def test_context_does_not_cross_section_headings():
    examples = list(iter_examples(["= Heading =", "First line here"]))
    assert [e.target for e in examples] == ["First", "line", "here"]


def test_count_words_only_credits_mid_sentence_capitals():
    counts, capitalised = count_words(["London is big. London is old. the end"])
    assert counts["london"] == 2
    # Both uses open a sentence, so neither counts as evidence of a proper noun.
    assert capitalised["london"] == 0
    counts, capitalised = count_words(["I visited London today"])
    assert capitalised["london"] == 1


# -- matching the context the daemon will actually have --------------------------


def test_trimming_is_off_at_zero_probability():
    example = Example(target="cat", left=["the", "big"], right=[], sentence_start=False)
    assert corpus.trim_to_runtime_context(example, random.Random(0), 0.0) is example


def test_trimming_keeps_the_words_nearest_the_gap():
    """What a daemon has is the most recent words, never the oldest ones."""
    example = Example(target="cat", left=["a", "b", "c", "d"], right=[], sentence_start=False)
    seen = set()
    for seed in range(60):
        trimmed = corpus.trim_to_runtime_context(example, random.Random(seed), 1.0)
        assert trimmed.left == example.left[len(example.left) - len(trimmed.left) :]
        seen.add(len(trimmed.left))
    # Every length from nothing to the whole window should be reachable.
    assert seen == {0, 1, 2, 3, 4}


def test_an_emptied_context_says_it_starts_a_sentence():
    """An empty context is what the daemon reports at the start of a session, and it
    calls that a sentence start, so a trimmed example has to agree."""
    example = Example(target="cat", left=["a"], right=[], sentence_start=False)
    emptied = [
        trimmed
        for seed in range(40)
        if not (trimmed := corpus.trim_to_runtime_context(example, random.Random(seed), 1.0)).left
    ]
    assert emptied and all(t.sentence_start for t in emptied)


def test_trimming_never_touches_the_target_or_the_right_context():
    example = Example(target="cat", left=["a", "b"], right=["sat"], sentence_start=True)
    trimmed = corpus.trim_to_runtime_context(example, random.Random(3), 1.0)
    assert trimmed.target == "cat"
    assert trimmed.right == ["sat"]
