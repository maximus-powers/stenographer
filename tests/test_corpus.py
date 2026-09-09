from model.corpus import count_words, iter_examples, tokenise


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
