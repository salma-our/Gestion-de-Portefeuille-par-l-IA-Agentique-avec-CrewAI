import json

import numpy as np
import pandas as pd
import pytest

from src import tools
from src.finance import annualized_volatility


def _fake_prices(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=250)
    close = pd.DataFrame(
        {
            "AAA": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 250)),
            "BBB": 50 * np.cumprod(1 + rng.normal(0.0003, 0.02, 250)),
        },
        index=idx,
    )
    return pd.concat({"Close": close}, axis=1)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(tools, "_download", lambda *a, **k: _fake_prices())


def test_allocation_tool_is_consistent(offline):
    out = json.loads(tools.calculate_optimal_allocation.run(tickers="AAA,BBB", budget=5000.0))
    allocs = out["allocations"]
    invested = sum(float(a["real_amount_invested"]) for a in allocs.values())
    assert out["total_invested"] == pytest.approx(invested, abs=0.05)
    assert out["total_invested"] + out["cash_remaining"] == pytest.approx(5000.0, abs=0.05)
    for a in allocs.values():
        assert a["shares_to_buy"] * float(a["current_price"]) == pytest.approx(
            float(a["real_amount_invested"]), abs=0.05
        )


def test_allocation_gives_lower_vol_asset_higher_weight(offline):
    out = json.loads(tools.calculate_optimal_allocation.run(tickers="AAA,BBB", budget=5000.0))
    w = {t: float(a["recommended_weight"].rstrip("%")) for t, a in out["allocations"].items()}
    assert w["AAA"] > w["BBB"]


def test_risk_tool_matches_finance_module(offline):
    out = json.loads(tools.analyze_portfolio_risk.run(tickers="AAA,BBB", period="1y"))
    returns = _fake_prices()["Close"]["AAA"].pct_change().dropna()
    expected = round(annualized_volatility(returns) * 100, 2)
    assert out["metrics_per_asset"]["AAA"]["annual_volatility"] == f"{expected}%"
    assert out["correlation_matrix"]["AAA"]["AAA"] == pytest.approx(1.0)


def test_tools_return_fallback_when_no_data(monkeypatch):
    monkeypatch.setattr(tools, "_download", lambda *a, **k: pd.DataFrame())
    assert tools.FALLBACK_MSG == tools.analyze_portfolio_risk.run(tickers="AAA", period="1y")
    assert tools.FALLBACK_MSG == tools.calculate_optimal_allocation.run(
        tickers="AAA", budget=1000.0
    )


def _long_prices(cols, seed=0, n=600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = pd.DataFrame(
        {
            c: 100 * np.cumprod(1 + rng.normal(0.0004, 0.008 * (i + 1), n))
            for i, c in enumerate(cols)
        },
        index=idx,
    )
    return pd.concat({"Close": close}, axis=1)


@pytest.fixture
def offline_backtest(monkeypatch):
    def fake(tickers, *a, **k):
        if isinstance(tickers, str):
            return _long_prices([tickers], seed=7)
        return _long_prices(tickers)

    monkeypatch.setattr(tools, "_download", fake)


def test_backtest_tool_returns_all_strategies_and_benchmark(offline_backtest):
    out = json.loads(tools.backtest_portfolio.run(tickers="AAA,BBB,CCC", years=3))
    assert set(out["strategies"]) == {
        "Equal weight",
        "Inverse volatility",
        "Min variance",
        "Max Sharpe",
        "Risk parity",
        "Benchmark",
    }
    row = out["strategies"]["Equal weight"]
    assert row["max_drawdown"].endswith("%") and isinstance(row["sharpe"], float)
    assert "benchmark" not in out


def test_backtest_tool_numbers_come_from_python(offline_backtest):
    summary, _ = tools.run_backtest(["AAA", "BBB", "CCC"])
    out = json.loads(tools.backtest_portfolio.run(tickers="AAA,BBB,CCC", years=3))
    expected = f"{round(summary.loc['Min variance', 'cagr'] * 100, 2)}%"
    assert out["strategies"]["Min variance"]["cagr"] == expected


def test_backtest_tool_single_ticker_skips_optimizers(offline_backtest):
    out = json.loads(tools.backtest_portfolio.run(tickers="AAA", years=3))
    assert set(out["strategies"]) == {"Equal weight", "Inverse volatility", "Benchmark"}


def test_backtest_tool_falls_back_on_short_history(monkeypatch):
    monkeypatch.setattr(tools, "_download", lambda *a, **k: _fake_prices())
    assert tools.backtest_portfolio.run(tickers="AAA,BBB", years=3) == tools.FALLBACK_MSG


def test_tool_outputs_are_recorded_for_guardrails(offline):
    tools.TOOL_OUTPUTS.clear()
    out = tools.calculate_optimal_allocation.run(tickers="AAA,BBB", budget=5000.0)
    assert tools.TOOL_OUTPUTS == [out]
