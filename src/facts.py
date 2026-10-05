"""Collect the numeric facts of a report by running the tools directly in Python."""

import json
from typing import Any

from src.report_builder import Facts
from src.tools import (
    analyze_portfolio_risk,
    analyze_stock,
    backtest_portfolio,
    calculate_optimal_allocation,
)


def _parse(output: str) -> dict[str, Any] | None:
    """Tool output -> dict, or None when the tool returned its 'data unavailable' message."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def collect_facts(tickers: list[str], budget: float, profile: str) -> Facts:
    """Run every tool once; outputs are recorded in TOOL_OUTPUTS as the source of truth."""
    joined = ",".join(tickers)
    return Facts(
        tickers=list(tickers),
        budget=budget,
        profile=profile,
        stocks={t: _parse(analyze_stock.run(ticker=t)) for t in tickers},
        risk=_parse(analyze_portfolio_risk.run(tickers=joined, period="1y")),
        allocation=_parse(calculate_optimal_allocation.run(tickers=joined, budget=budget)),
        backtest=_parse(backtest_portfolio.run(tickers=joined, years=3)),
    )
