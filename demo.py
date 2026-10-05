"""Run the three portfolio tools on live yfinance data (no LLM key needed)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tools import analyze_portfolio_risk, analyze_stock, calculate_optimal_allocation

TICKERS = "AAPL,MSFT,ASML.AS"
BUDGET = 10000.0


def show(title: str, raw: str) -> None:
    """Pretty-print a tool result; fail loudly if data is unavailable."""
    print(f"\n[{title}]")
    try:
        print(json.dumps(json.loads(raw), indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(raw)
        sys.exit(1)


def run_demo() -> None:
    show("stock_analysis AAPL", analyze_stock.run(ticker="AAPL"))
    show("portfolio_risk", analyze_portfolio_risk.run(tickers=TICKERS, period="1y"))
    show("portfolio_allocation", calculate_optimal_allocation.run(tickers=TICKERS, budget=BUDGET))


if __name__ == "__main__":
    run_demo()
