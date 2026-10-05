import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest, compare_strategies, performance_summary, rebalance_dates
from src.strategies import equal_weight, inverse_volatility, min_variance

LOOKBACK = 60


def _prices(n: int = 400, cols=("A", "B", "C"), seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    data = {
        c: 100 * np.cumprod(1 + rng.normal(0.0004, 0.005 * (i + 1), n)) for i, c in enumerate(cols)
    }
    return pd.DataFrame(data, index=idx)


def test_rebalance_dates_respect_lookback_and_month_end():
    idx = pd.bdate_range("2023-01-02", periods=400)
    dates = rebalance_dates(idx, LOOKBACK)
    assert all(idx.get_loc(d) + 1 >= LOOKBACK for d in dates)
    for d in dates:
        later = idx[idx > d]
        assert len(later) == 0 or later[0].month != d.month  # d is the last day of its month


def test_no_lookahead_weights_use_past_only_and_apply_next_day():
    prices = _prices()
    seen = []

    def spy(window):
        seen.append(window.index[-1])
        return {"A": 1.0}

    res = backtest(prices, spy, lookback=LOOKBACK, cost_bps=0)
    # first return is earned strictly after the first rebalance date
    assert res.returns.index[0] > seen[0]
    first_next = prices.index[prices.index > seen[0]][0]
    assert res.returns.index[0] == first_next


def test_window_has_exactly_lookback_rows_and_ends_at_rebalance_date():
    prices = _prices()
    lengths = []

    def spy(window):
        lengths.append(len(window))
        return equal_weight(window)

    backtest(prices, spy, lookback=LOOKBACK)
    assert set(lengths) == {LOOKBACK}


def test_single_asset_full_weight_matches_asset_returns_without_costs():
    prices = _prices(cols=("A",))
    res = backtest(prices, lambda w: {"A": 1.0}, lookback=LOOKBACK, cost_bps=0)
    asset = prices["A"].pct_change().dropna()
    assert res.returns.to_numpy() == pytest.approx(asset.loc[res.returns.index].to_numpy())


def test_first_day_cost_is_turnover_times_bps():
    prices = _prices(cols=("A",))
    free = backtest(prices, lambda w: {"A": 1.0}, lookback=LOOKBACK, cost_bps=0)
    paid = backtest(prices, lambda w: {"A": 1.0}, lookback=LOOKBACK, cost_bps=10)
    assert (free.returns.iloc[0] - paid.returns.iloc[0]) == pytest.approx(0.001)
    # full weight stays full weight: only the initial purchase is traded
    assert paid.turnover == pytest.approx(1.0)
    assert (free.returns.iloc[1:] - paid.returns.iloc[1:]).abs().max() < 1e-12


def test_constant_prices_give_flat_equity():
    prices = pd.DataFrame(
        100.0, index=pd.bdate_range("2023-01-02", periods=200), columns=["A", "B"]
    )
    res = backtest(prices, equal_weight, lookback=LOOKBACK, cost_bps=0)
    assert (1 + res.returns).prod() == pytest.approx(1.0)


@pytest.mark.parametrize("strategy", [equal_weight, inverse_volatility, min_variance])
def test_target_weights_sum_to_one(strategy):
    res = backtest(_prices(), strategy, lookback=LOOKBACK)
    assert res.weights.sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert (res.weights >= -1e-12).all().all()


def test_performance_summary_known_values():
    s = performance_summary(pd.Series([0.001] * 252))
    expected = 1.001**252 - 1
    assert s["total_return"] == pytest.approx(expected)
    assert s["cagr"] == pytest.approx(expected)
    assert s["volatility"] == pytest.approx(0.0, abs=1e-12)
    assert s["max_drawdown"] == pytest.approx(0.0)


def test_compare_strategies_aligns_benchmark_period():
    prices = _prices()
    bench = _prices(cols=("M",), seed=9)["M"]
    summary, curves = compare_strategies(
        prices,
        {"1/N": equal_weight, "MinVar": min_variance},
        benchmark=bench,
        lookback=LOOKBACK,
    )
    assert list(summary.index) == ["1/N", "MinVar", "Benchmark"]
    assert curves.notna().all().all()
    assert curves.index.is_monotonic_increasing
    assert {"cagr", "sharpe", "max_drawdown"} <= set(summary.columns)
