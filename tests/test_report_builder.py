import json
from datetime import datetime

import pytest

from src import pipeline
from src.guardrails import validate_report
from src.report_builder import MISSING, Facts, build_report, render_allocation, render_stocks

NOW = datetime(2026, 10, 5, 18, 9)
NARRATIVE = (
    "### Résumé exécutif\nPortefeuille concentré.\n\n### Conclusion\nSurveiller les résultats."
)

AAPL = {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "current_price": "333.52 USD",
    "variation_1d": "-0.09%",
    "variation_1m": "5.43%",
    "market_cap": 4865845624832,
    "pe_ratio": 38.24,
    "dividend_yield": 0.32,
    "52w_high": 345.34,
    "52w_low": 243.42,
    "avg_volume": 46009485,
    "beta": 1.07,
}
MSFT = {**AAPL, "ticker": "MSFT", "name": "Microsoft", "pe_ratio": "N/A", "beta": 0.9}
RISK = {
    "period_analyzed": "1y",
    "metrics_per_asset": {
        "AAPL": {
            "annual_volatility": "24.73%",
            "annual_return_estimate": "29.8%",
            "sharpe_ratio": 1.21,
            "max_drawdown": "-13.8%",
        },
        "MSFT": {
            "annual_volatility": "32.78%",
            "annual_return_estimate": "5.37%",
            "sharpe_ratio": 0.16,
            "max_drawdown": "-34.5%",
        },
    },
    "correlation_matrix": {
        "AAPL": {"AAPL": 1.0, "MSFT": 0.14},
        "MSFT": {"AAPL": 0.14, "MSFT": 1.0},
    },
}
ALLOCATION = {
    "base_currency": "EUR",
    "total_budget": 5000.0,
    "total_invested": 4780.16,
    "cash_remaining": 219.84,
    "method": "Risk-Parity (inverse volatility)",
    "allocations": {
        "AAPL": {
            "recommended_weight": "57.7%",
            "allocated_amount": "2885.0",
            "currency": "USD",
            "price_local": "374.38",
            "fx_rate_to_eur": 0.8909,
            "current_price": "333.52",
            "shares_to_buy": 8,
            "real_amount_invested": "2668.16",
        },
        "MSFT": {
            "recommended_weight": "42.3%",
            "allocated_amount": "2115.0",
            "currency": "EUR",
            "price_local": "528.0",
            "fx_rate_to_eur": 1.0,
            "current_price": "528.0",
            "shares_to_buy": 4,
            "real_amount_invested": "2112.0",
        },
    },
}
BACKTEST = {
    "period": "2024-11-01 to 2026-10-05",
    "assumptions": "long-only, monthly rebalance, 252-day estimation window, 10 bps cost",
    "strategies": {
        "Equal weight": {
            "total_return": "42.48%",
            "cagr": "20.38%",
            "volatility": "23.35%",
            "sharpe": 0.91,
            "max_drawdown": "-26.6%",
        },
        "Benchmark": {
            "total_return": "36.25%",
            "cagr": "17.59%",
            "volatility": "16.28%",
            "sharpe": 1.08,
            "max_drawdown": "-18.9%",
        },
    },
}


def _facts(**overrides) -> Facts:
    base = {
        "tickers": ["AAPL", "MSFT"],
        "budget": 5000.0,
        "profile": "moderate",
        "stocks": {"AAPL": AAPL, "MSFT": MSFT},
        "risk": RISK,
        "allocation": ALLOCATION,
        "backtest": BACKTEST,
    }
    return Facts(**{**base, **overrides})


def _truth(facts: Facts) -> list[str]:
    parts = [*facts.stocks.values(), facts.risk, facts.allocation, facts.backtest]
    return [json.dumps(p) for p in parts if p]


def test_built_report_passes_every_guardrail():
    facts = _facts()
    report = build_report(facts, NARRATIVE, NOW)
    result = validate_report(report, _truth(facts), facts.budget, NARRATIVE)
    assert result.passed, [c for c in result.checks if not c.passed]


def test_french_number_formatting_in_allocation():
    out = render_allocation(_facts())
    assert "2 885,00 €" in out and "57,7 %" in out and "219,84 €" in out
    assert "0,8909" in out and "Taux vers EUR" in out


def test_missing_values_are_reported_not_invented():
    out = render_stocks(_facts(stocks={"AAPL": AAPL, "MSFT": None}))
    msft_row = next(line for line in out.splitlines() if line.startswith("| MSFT"))
    assert msft_row.count(MISSING) == 10
    aapl_row = next(line for line in out.splitlines() if line.startswith("| AAPL"))
    assert MISSING not in aapl_row


def test_na_field_renders_as_missing():
    out = render_stocks(_facts())
    msft_row = next(line for line in out.splitlines() if line.startswith("| MSFT"))
    assert msft_row.count(MISSING) == 1  # only the P/E


def test_unavailable_tool_data_is_acknowledged_and_passes_guardrails():
    facts = _facts(allocation=None, backtest=None)
    report = build_report(facts, NARRATIVE, NOW)
    assert f"Allocation et cash restant : {MISSING}" in report and f"Backtest : {MISSING}" in report
    truth = [*_truth(facts), "⚠️ DATA UNAVAILABLE — Yahoo"]
    assert validate_report(report, truth, 5000.0, NARRATIVE).passed


def test_narrative_with_a_figure_fails_the_guardrails():
    facts = _facts()
    bad = NARRATIVE + "\nRendement attendu de 12,5 %."
    report = build_report(facts, bad, NOW)
    result = validate_report(report, _truth(facts), facts.budget, bad)
    assert not result.passed
    assert {c.name for c in result.checks if not c.passed} >= {"narrative_without_figures"}


def test_empty_narrative_fails_required_sections():
    facts = _facts()
    report = build_report(facts, "", NOW)
    result = validate_report(report, _truth(facts), facts.budget, "")
    assert not result.passed and f"Commentaire : {MISSING}" in report


def test_report_has_all_numbered_sections_and_disclaimer():
    report = build_report(_facts(), NARRATIVE, NOW)
    for title in ("## 1. Données", "## 2. Analyse du risque", "## 3. Allocation", "## 4. Backtest"):
        assert title in report
    assert "05/10/2026 18:09" in report and "Ne constitue pas un conseil" in report


def test_finalize_report_appends_validation_footer(monkeypatch):
    facts = _facts()
    monkeypatch.setattr(pipeline, "collect_facts", lambda *a, **k: facts)
    pipeline.TOOL_OUTPUTS.clear()
    pipeline.TOOL_OUTPUTS.extend(_truth(facts))
    report, validation = pipeline.finalize_report(
        NARRATIVE, ["AAPL", "MSFT"], 5000.0, "moderate", NOW
    )
    assert validation.passed
    assert "Validation automatique : VALIDÉ" in report


@pytest.mark.parametrize("missing", ["risk", "allocation", "backtest"])
def test_each_numeric_block_degrades_gracefully(missing):
    report = build_report(_facts(**{missing: None}), NARRATIVE, NOW)
    assert MISSING in report
