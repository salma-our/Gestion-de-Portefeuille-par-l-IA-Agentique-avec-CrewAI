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
    monkeypatch.setattr(tools, "_currency", lambda t: "EUR")


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


def test_allocation_converts_prices_to_base_currency(monkeypatch):
    monkeypatch.setattr(tools, "_download", lambda *a, **k: _fake_prices())
    monkeypatch.setattr(tools, "_currency", lambda t: "USD")
    monkeypatch.setattr(tools, "_fx_to_base", lambda c, b="EUR": 0.5)
    out = json.loads(tools.calculate_optimal_allocation.run(tickers="AAA,BBB", budget=5000.0))
    for a in out["allocations"].values():
        assert a["currency"] == "USD" and a["fx_rate_to_eur"] == 0.5
        assert float(a["current_price"]) == pytest.approx(float(a["price_local"]) * 0.5, abs=0.01)
    assert out["total_invested"] <= 5000.0


def test_allocation_falls_back_when_fx_unavailable(monkeypatch):
    monkeypatch.setattr(tools, "_download", lambda *a, **k: _fake_prices())
    monkeypatch.setattr(tools, "_currency", lambda t: "USD")

    def no_fx(currency, base="EUR"):
        raise ValueError("FX rate unavailable")

    monkeypatch.setattr(tools, "_fx_to_base", no_fx)
    assert (
        tools.calculate_optimal_allocation.run(tickers="AAA", budget=1000.0) == tools.FALLBACK_MSG
    )


# ---- news sentiment tool ----
def _item(title, days_ago=1, publisher="Wire"):
    from datetime import UTC, datetime, timedelta

    from src.sentiment import NewsItem

    return NewsItem(title, publisher, (datetime.now(UTC) - timedelta(days=days_ago)).date())


def test_news_tool_scores_and_aggregates(monkeypatch):
    items = [
        _item("Apple profit surges", 1),
        _item("Apple faces antitrust probe", 2),
        _item("Apple event", 3),
    ]
    monkeypatch.setattr(tools, "_fetch_news", lambda *a, **k: items)
    out = json.loads(tools.analyze_news_sentiment.run(tickers="AAPL", max_headlines=2))
    aapl = out["tickers"]["AAPL"]
    assert aapl["headlines_count"] == 3 and aapl["mean_score"] == 0.0
    assert aapl["positive_share"] == "33.3%" and aapl["negative_share"] == "33.3%"
    assert [h["score"] for h in aapl["latest"]] == [1.0, -1.0]  # only the 2 latest, newest first


def test_news_tool_reports_na_without_headlines(monkeypatch):
    monkeypatch.setattr(tools, "_fetch_news", lambda *a, **k: [])
    aapl = json.loads(tools.analyze_news_sentiment.run(tickers="AAPL"))["tickers"]["AAPL"]
    assert aapl["headlines_count"] == 0 and aapl["mean_score"] == "N/A" and aapl["latest"] == []


def test_yahoo_news_parsing(monkeypatch):
    class FakeSearch:
        def __init__(self, ticker, news_count):
            self.news = [
                {"title": "Good news", "publisher": "WSJ", "providerPublishTime": 1_790_000_000},
                {"title": None, "providerPublishTime": 1_790_000_000},
                {"title": "No date"},
            ]

    monkeypatch.setattr(tools.yf, "Search", FakeSearch)
    items = tools._yahoo_news("AAPL", 5)
    assert [(i.title, i.publisher) for i in items] == [("Good news", "WSJ")]


def test_yahoo_news_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(tools.yf, "Search", boom)
    assert tools._yahoo_news("AAPL", 5) == []


def test_rss_news_parsing_and_publisher_suffix(monkeypatch):
    xml = (
        b"<rss><channel><item><title>Apple soars - Barchart</title>"
        b"<pubDate>Sat, 03 Oct 2026 18:30:02 GMT</pubDate><source>Barchart</source></item>"
        b"<item><title>No date</title></item></channel></rss>"
    )

    class FakeResponse:
        content = xml

        def raise_for_status(self):
            pass

    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse())
    items = tools._rss_news("AAPL", 5)
    assert len(items) == 1 and items[0].title == "Apple soars" and items[0].publisher == "Barchart"
    assert items[0].published.isoformat() == "2026-10-03"


def test_fetch_news_falls_back_to_rss_filters_old_and_dedupes(monkeypatch):
    monkeypatch.setattr(tools, "_yahoo_news", lambda *a, **k: [])
    rss = [_item("Fresh", 1), _item("Fresh", 2), _item("Ancient", 90), _item("Older fresh", 5)]
    monkeypatch.setattr(tools, "_rss_news", lambda *a, **k: rss)
    titles = [i.title for i in tools._fetch_news("AAPL", limit=10, days=30)]
    assert titles == ["Fresh", "Older fresh"]
