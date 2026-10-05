import math

import pandas as pd
import pytest

from src.finance import (
    allocate,
    annualized_return,
    annualized_volatility,
    inverse_volatility_weights,
    max_drawdown,
    sharpe_ratio,
)


def test_volatility_of_two_returns():
    r = pd.Series([0.01, 0.03])
    # sample std of [0.01, 0.03] = 0.0141421...
    assert annualized_volatility(r) == pytest.approx(0.0141421356 * math.sqrt(252))


def test_volatility_of_constant_returns_is_zero():
    assert annualized_volatility(pd.Series([0.01] * 10)) == pytest.approx(0.0, abs=1e-12)


def test_annualized_return_is_mean_times_252():
    assert annualized_return(pd.Series([0.01, 0.03])) == pytest.approx(0.02 * 252)


def test_sharpe_ratio_and_zero_volatility():
    assert sharpe_ratio(0.2, 0.1) == pytest.approx(2.0)
    assert sharpe_ratio(0.2, 0.0) == 0.0


def test_max_drawdown_known_curve():
    # cumulative: 1.1 -> 0.55 -> 0.66 ; worst drop from peak 1.1 to 0.55 = -50%
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_max_drawdown_monotonic_gain_is_zero():
    assert max_drawdown(pd.Series([0.01, 0.02, 0.03])) == 0.0


def test_inverse_volatility_weights():
    w = inverse_volatility_weights({"A": 0.2, "B": 0.4})
    assert w["A"] == pytest.approx(2 / 3)
    assert w["B"] == pytest.approx(1 / 3)
    assert sum(w.values()) == pytest.approx(1.0)


def test_inverse_volatility_ignores_non_positive_and_empty():
    assert inverse_volatility_weights({"A": 0.0, "B": -1.0}) == {}
    assert inverse_volatility_weights({}) == {}
    assert list(inverse_volatility_weights({"A": 0.0, "B": 0.5})) == ["B"]


def test_allocate_whole_shares_and_cash():
    positions, cash = allocate({"A": 0.5, "B": 0.5}, {"A": 30.0, "B": 70.0}, 1000.0)
    assert positions["A"].shares == 16 and positions["A"].invested == pytest.approx(480.0)
    assert positions["B"].shares == 7 and positions["B"].invested == pytest.approx(490.0)
    assert cash == pytest.approx(30.0)


def test_allocate_never_exceeds_budget():
    positions, cash = allocate({"A": 0.6, "B": 0.4}, {"A": 333.51, "B": 524.86}, 5000.0)
    assert cash >= 0
    assert sum(p.invested for p in positions.values()) + cash == pytest.approx(5000.0)


def test_allocate_missing_price_buys_nothing():
    positions, cash = allocate({"A": 1.0}, {}, 1000.0)
    assert positions["A"].shares == 0 and cash == 1000.0
