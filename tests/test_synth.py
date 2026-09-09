import random

import pytest

from model.synth import (
    POLICIES,
    SynthConfig,
    is_typable,
    same_finger_pairs,
    sample_chord,
)
from model.vocab import distinct_letters

WORDS = ["there", "tomorrow", "because", "stenographer", "letter", "the", "of", "banana"]


@pytest.mark.parametrize("word", WORDS)
@pytest.mark.parametrize("policy", sorted(POLICIES))
@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_generated_chord_is_always_a_subset_of_the_word(word, policy, size):
    # This is the contract candidate generation depends on: a word is only a candidate
    # when it contains every chord letter, so a chord outside its word is unreachable.
    rng = random.Random(0)
    config = SynthConfig()
    result = sample_chord(word, rng, config, policy=policy, size=size)
    assert result is not None
    chord, _ = result
    assert set(chord) <= set(distinct_letters(word))


@pytest.mark.parametrize("word", WORDS)
@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_chord_size_is_capped_by_available_letters(word, policy):
    rng = random.Random(1)
    available = len(distinct_letters(word))
    chord, _ = sample_chord(word, rng, SynthConfig(), policy=policy, size=5)
    assert len(chord) == min(5, available)


def test_chord_letters_are_sorted_and_distinct():
    rng = random.Random(2)
    chord, _ = sample_chord("letter", rng, SynthConfig(), policy="prefix", size=4)
    assert chord == "".join(sorted(set(chord)))


def test_words_without_two_distinct_letters_cannot_be_chorded():
    rng = random.Random(3)
    assert sample_chord("aaa", rng, SynthConfig()) is None
    assert sample_chord("a", rng, SynthConfig()) is None


def test_noise_is_off_by_default_so_targets_stay_reachable():
    config = SynthConfig()
    assert config.substitution_noise == 0.0
    rng = random.Random(4)
    for _ in range(200):
        chord, policy = sample_chord("because", rng, config)
        assert set(chord) <= set("becaus")
        assert policy in POLICIES


def test_noise_when_enabled_lands_off_target_and_is_labelled():
    # The label matters: accuracy tables must never credit a policy for a noisy chord.
    rng = random.Random(5)
    config = SynthConfig(substitution_noise=1.0)
    off_target = 0
    for _ in range(100):
        chord, policy = sample_chord("because", rng, config, policy="prefix", size=3)
        assert policy == "noisy"
        if not set(chord) <= set("becaus"):
            off_target += 1
    assert off_target == 100


def test_prefix_policy_takes_the_leading_letters():
    rng = random.Random(6)
    chord, _ = sample_chord("because", rng, SynthConfig(), policy="prefix", size=3)
    assert chord == "".join(sorted("bec"))


def test_skeleton_policy_drops_vowels_after_the_first_letter():
    rng = random.Random(7)
    chord, _ = sample_chord("because", rng, SynthConfig(), policy="skeleton", size=3)
    assert chord == "".join(sorted("bcs"))


def test_same_finger_detection():
    assert same_finger_pairs("for") == 1  # f and r are both left index
    assert same_finger_pairs("adn") == 0
    assert is_typable("adn")
    assert not is_typable("for")
