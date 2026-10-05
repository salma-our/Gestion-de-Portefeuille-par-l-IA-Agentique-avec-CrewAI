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
