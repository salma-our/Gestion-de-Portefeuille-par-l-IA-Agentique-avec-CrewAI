import pytest

from src.sentiment import aggregate, score_headline


def test_positive_headline():
    assert score_headline("Apple earnings beat estimates as profit surges") == 1.0


def test_negative_headline():
    assert score_headline("Apple Is Dealing With New Problems Now") == -1.0
    assert score_headline("Shares plunge after antitrust probe") == -1.0


def test_mixed_headline_is_between_extremes():
    # 1 positive (gains), 2 negative (risks, concerns) -> (1 - 2) / 3
    assert score_headline("Stock gains despite risks and concerns") == pytest.approx(-1 / 3)


def test_neutral_or_empty_headline_scores_zero():
    assert score_headline("Apple to hold event on Tuesday") == 0.0
    assert score_headline("") == 0.0


def test_negation_flips_polarity():
    assert score_headline("Company does not beat estimates") == -1.0
    assert score_headline("Profit won't fall") == 1.0
    assert score_headline("No growth expected") == -1.0


def test_negation_only_reaches_two_tokens_back():
    assert score_headline("not a single analyst thinks profit will rise") == 1.0


def test_case_and_punctuation_are_ignored():
    assert score_headline("RECORD Profits!!!") == 1.0


def test_aggregate_known_values():
    agg = aggregate([1.0, -1.0, 0.0, 0.5])
    assert agg["count"] == 4
    assert agg["mean"] == pytest.approx(0.125)
    assert agg["positive_share"] == pytest.approx(0.5)
    assert agg["negative_share"] == pytest.approx(0.25)


def test_aggregate_without_headlines():
    assert aggregate([]) == {
        "count": 0,
        "mean": None,
        "positive_share": None,
        "negative_share": None,
    }
