"""
tools.py — CrewAI 0.28 tools for portfolio analysis
Uses the @tool decorator from crewai.tools
Integrates yfinance with robust error handling and retry logic.
"""

import functools
import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

import pandas as pd
import requests
import yfinance as yf
from crewai.tools import tool

from src import strategies
from src.backtest import compare_strategies
from src.finance import (
    allocate,
    annualized_return,
    annualized_volatility,
    inverse_volatility_weights,
    max_drawdown,
    sharpe_ratio,
)
from src.sentiment import NewsItem, aggregate, score_headline

logger = logging.getLogger(__name__)

# Raw outputs of every tool call in the current run; used to validate the final report.
TOOL_OUTPUTS: list[str] = []

NEWS_WINDOW_DAYS = 30
NEWS_FETCH_LIMIT = 10
BASE_CURRENCY = "EUR"
BENCHMARK = "^GSPC"
BACKTEST_LOOKBACK = 252
MIN_BACKTEST_ROWS = BACKTEST_LOOKBACK + 42

# Errors that mean 'the data is unusable', as opposed to programming bugs.
DATA_ERRORS = (ValueError, KeyError, IndexError, ZeroDivisionError)

FALLBACK_MSG = (
    "⚠️ DATA UNAVAILABLE — Yahoo Finance data temporarily inaccessible. "
    "Cannot produce reliable analysis without real data. "
    "Report 'donnée indisponible' instead of estimating or inventing. "
    "NEVER invent metrics — better incomplete+honest than invented+false."
)


def _record(fn):
    """Append the tool's raw output to TOOL_OUTPUTS (source of truth for guardrails)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        output = fn(*args, **kwargs)
        TOOL_OUTPUTS.append(output)
        return output

    return wrapper


def _download(tickers, period: str = "1y", retries: int = 3) -> pd.DataFrame:
    """yf.download with retry and linear backoff; empty DataFrame if all attempts fail."""
    wait = 4
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers, period=period, auto_adjust=True, progress=False, threads=False
            )
        except Exception as exc:  # noqa: BLE001 - yfinance/curl raise many unrelated types
            logger.warning(
                "yfinance download failed (attempt %d/%d): %s", attempt + 1, retries, exc
            )
        else:
            if not raw.empty:
                return raw
            logger.warning("yfinance returned no data (attempt %d/%d)", attempt + 1, retries)
        time.sleep(wait * (attempt + 1))
    return pd.DataFrame()


def _currency(ticker: str) -> str:
    """Trading currency of a ticker as reported by Yahoo (for example USD, EUR, GBp)."""
    try:
        return str(yf.Ticker(ticker).fast_info["currency"])
    except Exception as exc:  # noqa: BLE001 - yfinance raises many unrelated types
        raise ValueError(f"Currency unavailable for {ticker}: {exc}") from exc


def _fx_to_base(currency: str, base: str = BASE_CURRENCY) -> float:
    """Units of `base` per unit of `currency`, from the latest Yahoo FX close."""
    scale = 1.0
    if currency == "GBp":  # London quotes in pence
        currency, scale = "GBP", 0.01
    if currency == base:
        return scale
    pair = f"{currency}{base}=X"
    data = _extract_close(_download(pair, period="5d"), pair)
    if pair not in data.columns or data[pair].dropna().empty:
        raise ValueError(f"FX rate unavailable for {pair}")
    return float(data[pair].dropna().iloc[-1]) * scale


def _extract_close(raw, tickers):
    """Extract Close column regardless of ticker count."""
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            data = raw["Close"]
        else:
            return pd.DataFrame()
    elif "Close" in raw.columns:
        data = raw[["Close"]]
        name = tickers if isinstance(tickers, str) else (tickers[0] if len(tickers) == 1 else None)
        if name:
            data.columns = [name]
    else:
        data = raw

    if isinstance(data, pd.Series):
        name = tickers if isinstance(tickers, str) else tickers[0]
        data = data.to_frame(name=name)

    return data


# Tool 1: Stock analysis
@tool("stock_analysis")
@_record
def analyze_stock(ticker: str) -> str:
    """
    Analyze a stock ticker: current price, variations, fundamentals (P/E, dividend, market cap).
    Provides key data to evaluate the stock.

    Args:
        ticker: Stock symbol, e.g., AAPL, MSFT, BNP.PA

    Returns:
        JSON string with stock analysis data
    """
    t = ticker.upper().strip()
    try:
        raw = _download(t, period="1mo")
        data = _extract_close(raw, t)

        if data.empty or t not in data.columns:
            return FALLBACK_MSG

        prices = data[t].dropna()
        if len(prices) < 2:
            return FALLBACK_MSG

        current_price = round(float(prices.iloc[-1]), 2)
        prev_price = round(float(prices.iloc[-2]), 2)
        start_price = round(float(prices.iloc[0]), 2)
        variation_1d = (
            round(((current_price - prev_price) / prev_price) * 100, 2) if prev_price else 0
        )
        variation_1m = (
            round(((current_price - start_price) / start_price) * 100, 2) if start_price else 0
        )

        # Fundamentals via .info (best-effort)
        info = {}
        try:
            time.sleep(1)
            info = yf.Ticker(t).info or {}
        except Exception as exc:  # noqa: BLE001 - fundamentals are best-effort
            logger.warning("Could not fetch fundamentals for %s: %s", t, exc)

        result = {
            "ticker": t,
            "name": info.get("longName", t),
            "sector": info.get("sector", "N/A"),
            "current_price": f"{current_price} {info.get('currency', 'USD')}",
            "variation_1d": f"{variation_1d}%",
            "variation_1m": f"{variation_1m}%",
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
            "dividend_yield": info.get("dividendYield", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
            "avg_volume": info.get("averageVolume", "N/A"),
            "beta": round(info.get("beta", 0), 2) if info.get("beta") else "N/A",
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except DATA_ERRORS:
        logger.exception("Tool failed on unusable data")
        return FALLBACK_MSG


# Tool 2: Portfolio risk analysis
@tool("portfolio_risk")
@_record
def analyze_portfolio_risk(tickers: str, period: str = "1y") -> str:
    """
    Calculate portfolio risk metrics: annual volatility, Sharpe ratio, max drawdown, correlation.

    Args:
        tickers: Stock symbols separated by commas, e.g., AAPL,MSFT
        period: Time period (1mo, 3mo, 6mo, 1y, 2y)

    Returns:
        JSON string with risk metrics
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    try:
        raw = _download(ticker_list, period=period)
        data = _extract_close(raw, ticker_list)

        if data.empty:
            return FALLBACK_MSG

        returns = data.pct_change().dropna()
        metrics = {}

        for t in ticker_list:
            if t not in returns.columns:
                continue
            r = returns[t].dropna()
            if r.empty:
                continue
            vol = annualized_volatility(r)
            rend = annualized_return(r)
            metrics[t] = {
                "annual_volatility": f"{round(vol * 100, 2)}%",
                "annual_return_estimate": f"{round(rend * 100, 2)}%",
                "sharpe_ratio": round(sharpe_ratio(rend, vol), 2),
                "max_drawdown": f"{round(max_drawdown(r) * 100, 2)}%",
            }

        available = [t for t in ticker_list if t in returns.columns]
        correlation = returns[available].corr().round(2).to_dict() if len(available) > 1 else {}

        result = {
            "period_analyzed": period,
            "metrics_per_asset": metrics,
            "correlation_matrix": correlation if correlation else "N/A (single asset)",
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except DATA_ERRORS:
        logger.exception("Tool failed on unusable data")
        return FALLBACK_MSG


# Tool 3: Optimal allocation
@tool("portfolio_allocation")
@_record
def calculate_optimal_allocation(tickers: str, budget: float = 10000.0) -> str:
    """
    Calculate optimal portfolio allocation (risk-parity) based on budget and profile.
    Returns recommended weights and number of shares to buy.

    Args:
        tickers: Stock symbols separated by commas, e.g., AAPL,MSFT
        budget: Total investment budget in EUR; prices are converted from their trading currency

    Returns:
        JSON string with allocation recommendations
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    try:
        raw = _download(ticker_list, period="6mo")
        data = _extract_close(raw, ticker_list)

        if data.empty:
            return FALLBACK_MSG

        current_prices: dict[str, float] = {}
        volatilities: dict[str, float] = {}

        for t in ticker_list:
            if t not in data.columns:
                continue
            col = data[t].dropna()
            if col.empty:
                continue
            current_prices[t] = round(float(col.iloc[-1]), 2)
            vol = annualized_volatility(col.pct_change().dropna())
            if vol > 0:
                volatilities[t] = vol

        weights = {t: round(w, 4) for t, w in inverse_volatility_weights(volatilities).items()}
        if not weights:
            return FALLBACK_MSG

        currencies = {t: _currency(t) for t in weights}
        fx = {c: _fx_to_base(c) for c in set(currencies.values())}
        base_prices = {t: round(current_prices[t] * fx[currencies[t]], 2) for t in weights}

        positions, cash = allocate(weights, base_prices, budget)
        allocations = {
            t: {
                "recommended_weight": f"{round(p.weight * 100, 1)}%",
                "allocated_amount": str(round(p.target_amount, 2)),
                "currency": currencies[t],
                "price_local": str(current_prices[t]),
                "fx_rate_to_eur": round(fx[currencies[t]], 4),
                "current_price": str(p.price),
                "shares_to_buy": p.shares,
                "real_amount_invested": str(round(p.invested, 2)),
            }
            for t, p in positions.items()
        }
        total_invested = budget - cash

        result = {
            "base_currency": BASE_CURRENCY,
            "total_budget": budget,
            "total_invested": round(total_invested, 2),
            "cash_remaining": round(budget - total_invested, 2),
            "method": "Risk-Parity (inverse volatility)",
            "allocations": allocations,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except DATA_ERRORS:
        logger.exception("Tool failed on unusable data")
        return FALLBACK_MSG


def run_backtest(
    tickers: list[str], years: int = 3, benchmark: str = BENCHMARK
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download prices and compare allocation strategies; returns (summary, growth curves)."""
    prices = _extract_close(_download(tickers, period=f"{years}y"), tickers)
    prices = prices[[t for t in tickers if t in prices.columns]].dropna()
    if len(prices) < MIN_BACKTEST_ROWS:
        raise ValueError(f"Not enough price history for a backtest ({len(prices)} rows)")

    n = prices.shape[1]
    candidates: dict[str, strategies.Strategy] = {
        "Equal weight": strategies.equal_weight,
        "Inverse volatility": strategies.inverse_volatility,
    }
    if n > 1:
        cap = max(0.5, 1.2 / n)
        candidates["Min variance"] = lambda w: strategies.min_variance(w, cap)
        candidates["Max Sharpe"] = lambda w: strategies.max_sharpe(w, cap)
        candidates["Risk parity"] = lambda w: strategies.risk_parity(w, cap)

    bench_prices = _extract_close(_download(benchmark, period=f"{years}y"), benchmark)
    bench = bench_prices[benchmark].dropna() if benchmark in bench_prices.columns else None
    return compare_strategies(prices, candidates, bench, lookback=BACKTEST_LOOKBACK)


# Tool 4: Backtest
@tool("portfolio_backtest")
@_record
def backtest_portfolio(tickers: str, years: int = 3) -> str:
    """
    Backtest allocation strategies (equal weight, inverse volatility, min variance, max Sharpe,
    risk parity) on historical prices with monthly rebalancing and transaction costs, and compare
    them with the S&P 500. All figures are computed by Python, never estimated.

    Args:
        tickers: Stock symbols separated by commas, e.g., AAPL,MSFT
        years: Years of history to download (default 3; the first year is the estimation window)

    Returns:
        JSON string with performance metrics per strategy
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    try:
        summary, curves = run_backtest(ticker_list, years)
    except DATA_ERRORS:
        logger.exception("Backtest failed on unusable data")
        return FALLBACK_MSG

    pct = ("total_return", "cagr", "volatility", "max_drawdown")
    strategies_out = {
        name: {
            k: (f"{round(v * 100, 2)}%" if k in pct else round(float(v), 2)) for k, v in row.items()
        }
        for name, row in summary.iterrows()
    }
    result = {
        "period": f"{curves.index[0].date()} to {curves.index[-1].date()}",
        "assumptions": (
            "long-only, monthly rebalance, 252-day estimation window, 10 bps transaction "
            f"cost, benchmark S&P 500 ({BENCHMARK}), returns in local currencies "
            "without FX adjustment"
        ),
        "strategies": strategies_out,
    }
    if "Benchmark" not in summary.index:
        result["benchmark"] = "donnée indisponible"
    return json.dumps(result, ensure_ascii=False, indent=2)


def _yahoo_news(ticker: str, limit: int) -> list[NewsItem]:
    """Headlines from Yahoo's search endpoint (Ticker.news is currently empty)."""
    try:
        raw = yf.Search(ticker, news_count=limit).news
    except Exception as exc:  # noqa: BLE001 - yfinance raises many unrelated types
        logger.warning("Yahoo news failed for %s: %s", ticker, exc)
        return []
    items = []
    for entry in raw:
        title, ts = entry.get("title"), entry.get("providerPublishTime")
        if title and ts:
            published = datetime.fromtimestamp(ts, UTC).date()
            items.append(NewsItem(title, entry.get("publisher") or "Yahoo Finance", published))
    return items


def _rss_news(ticker: str, limit: int) -> list[NewsItem]:
    """Fallback: Google News RSS search for the ticker."""
    params = {"q": f"{ticker} stock", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    try:
        response = requests.get("https://news.google.com/rss/search", params=params, timeout=10)
        response.raise_for_status()
        entries = ET.fromstring(response.content).findall("./channel/item")
    except (requests.RequestException, ET.ParseError) as exc:
        logger.warning("RSS news failed for %s: %s", ticker, exc)
        return []
    items = []
    for entry in entries[:limit]:
        title, pub_date = entry.findtext("title"), entry.findtext("pubDate")
        source = entry.findtext("source") or "Google News"
        if not title or not pub_date:
            continue
        title = title.removesuffix(f" - {source}")
        published = parsedate_to_datetime(pub_date).astimezone(UTC).date()
        items.append(NewsItem(title, source, published))
    return items


def _fetch_news(ticker: str, limit: int = NEWS_FETCH_LIMIT, days: int = NEWS_WINDOW_DAYS):
    """Recent unique headlines (newest first), Yahoo first and RSS as a fallback."""
    cutoff = (datetime.now(UTC) - timedelta(days=days)).date()
    items = _yahoo_news(ticker, limit) or _rss_news(ticker, limit)
    unique = {item.title: item for item in items if item.published >= cutoff}
    return sorted(unique.values(), key=lambda i: i.published, reverse=True)[:limit]


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{round(value * 100, 1)}%"


# Tool 5: News sentiment
@tool("news_sentiment")
@_record
def analyze_news_sentiment(tickers: str, max_headlines: int = 3) -> str:
    """
    Fetch recent news headlines per ticker and score their sentiment with a finance lexicon.
    Scores are computed by Python in [-1, 1]; the language model must not score or recount.

    Args:
        tickers: Stock symbols separated by commas, e.g., AAPL,MSFT
        max_headlines: Number of latest headlines to list per ticker

    Returns:
        JSON string with the aggregate score and latest headlines for each ticker
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    per_ticker = {}
    for ticker in ticker_list:
        scored = [(item, score_headline(item.title)) for item in _fetch_news(ticker)]
        agg = aggregate([score for _, score in scored])
        per_ticker[ticker] = {
            "headlines_count": agg["count"],
            "mean_score": "N/A" if agg["mean"] is None else round(agg["mean"], 2),
            "positive_share": _pct(agg["positive_share"]),
            "negative_share": _pct(agg["negative_share"]),
            "latest": [
                {
                    "title": item.title,
                    "publisher": item.publisher,
                    "date": item.published.isoformat(),
                    "score": round(score, 2),
                }
                for item, score in scored[:max_headlines]
            ],
        }
    result = {
        "window_days": NEWS_WINDOW_DAYS,
        "scoring": "finance-lexicon headline score in [-1, 1]; mean over headlines",
        "tickers": per_ticker,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)
