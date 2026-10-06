"""Deterministic offline market used by the evaluation scenarios (no network, no LLM)."""

import zlib
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import date
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import tools
from src.sentiment import NewsItem

PERIOD_ROWS = {"5d": 5, "1mo": 22, "6mo": 126, "1y": 252, "2y": 504, "3y": 756}
FX_TO_EUR = {"EUR": 1.0, "USD": 0.9, "GBP": 1.15, "GBp": 0.0115}


def _prices(ticker: str, index: pd.DatetimeIndex) -> pd.Series:
    seed = zlib.crc32(ticker.encode())
    rng = np.random.default_rng(seed)
    daily_vol = 0.008 + (seed % 7) * 0.002
    start = 20 + seed % 400
    return pd.Series(start * np.cumprod(1 + rng.normal(0.0004, daily_vol, len(index))), index=index)


class FakeTicker:
    """Stand-in for yfinance.Ticker exposing only `.info`."""

    invalid: frozenset[str] = frozenset()

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    @property
    def info(self) -> dict[str, Any]:
        if self.ticker in self.invalid:
            return {}
        return {
            "longName": f"{self.ticker} Corp.",
            "sector": "Technology",
            "currency": "EUR",
            "marketCap": 1_000_000_000,
            "trailingPE": 21.5,
            "dividendYield": 0.5,
            "fiftyTwoWeekHigh": 500.0,
            "fiftyTwoWeekLow": 300.0,
            "averageVolume": 1_000_000,
            "beta": 1.1,
        }


@contextmanager
def patched_market(
    invalid: frozenset[str] = frozenset(),
    currencies: dict[str, str] | None = None,
    history_days: int = 800,
    news_mode: str = "default",
) -> Iterator[None]:
    """Replace every network access of src.tools with deterministic synthetic data."""
    currencies = currencies or {}
    master = pd.bdate_range(end="2026-10-02", periods=history_days)

    def download(tickers: Any, period: str = "1y", retries: int = 3) -> pd.DataFrame:
        names = [tickers] if isinstance(tickers, str) else list(tickers)
        valid = [t for t in names if t not in invalid]
        if not valid:
            return pd.DataFrame()
        index = master[-PERIOD_ROWS.get(period, 252) :]
        frame = pd.DataFrame({t: _prices(t, index) for t in valid})
        return pd.concat({"Close": frame}, axis=1)

    def fetch_news(ticker: str, limit: int = 10, days: int = 30) -> list[NewsItem]:
        if news_mode == "none" or ticker in invalid:
            return []
        day = date(2026, 10, 1)
        if news_mode == "negative":
            titles = [
                f"{ticker} plunges after fraud probe",
                f"{ticker} faces lawsuit and recall",
                f"{ticker} warns of weak demand",
            ]
        else:
            titles = [f"{ticker} profit beats estimates", f"{ticker} faces regulatory probe"]
        return [NewsItem(title, "Test Wire", day) for title in titles]

    class Ticker(FakeTicker):
        pass

    Ticker.invalid = invalid

    with ExitStack() as stack:
        stack.enter_context(patch.object(tools, "_download", download))
        stack.enter_context(patch.object(tools, "_fetch_news", fetch_news))
        stack.enter_context(patch.object(tools, "_currency", lambda t: currencies.get(t, "EUR")))
        stack.enter_context(patch.object(tools, "_fx_to_base", lambda c, b="EUR": FX_TO_EUR[c]))
        stack.enter_context(patch.object(tools.yf, "Ticker", Ticker))
        stack.enter_context(patch.object(tools.time, "sleep", lambda s: None))
        yield
