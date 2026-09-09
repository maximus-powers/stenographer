import pytest

from model.speed_model import (
    SpeedParams,
    break_even_top1,
    chording_cost,
    typing_cost,
)


def test_typing_cost_counts_the_trailing_space():
    params = SpeedParams(keystroke=0.2)
    assert typing_cost(5, params) == pytest.approx(1.2)


def test_perfect_accuracy_costs_exactly_one_chord():
    params = SpeedParams()
    assert chording_cost(5, 1.0, 1.0, params) == pytest.approx(params.chord)


def test_cost_falls_as_accuracy_rises():
    params = SpeedParams()
    worse = chording_cost(6, 0.4, 0.6, params)
    better = chording_cost(6, 0.8, 0.92, params)
    assert better < worse


def test_break_even_is_the_point_where_chording_matches_typing():
    params = SpeedParams()
    for length in (4, 5, 6, 7):
        top3 = 0.92
        needed = break_even_top1(length, top3, params)
        if 0.0 < needed < 1.0:
            assert chording_cost(length, needed, top3, params) == pytest.approx(
                typing_cost(length, params), abs=1e-9
            )


def test_longer_words_break_even_at_lower_accuracy():
    params = SpeedParams()
    assert break_even_top1(4, 0.92, params) > break_even_top1(7, 0.92, params)


def test_slower_chording_demands_higher_accuracy():
    quick = break_even_top1(5, 0.92, SpeedParams(chord=0.6))
    slow = break_even_top1(5, 0.92, SpeedParams(chord=0.9))
    assert slow > quick
