"""Walk-forward backtest: weights are computed on past data only and applied the next day."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.finance import TRADING_DAYS, annualized_return, annualized_volatility, max_drawdown
from src.finance import sharpe_ratio as _sharpe
from src.strategies import Strategy


@dataclass(frozen=True)
class BacktestResult:
    """Net daily returns, target weights at each rebalance date, and total turnover."""

    returns: pd.Series
    weights: pd.DataFrame
    turnover: float


def rebalance_dates(index: pd.DatetimeIndex, lookback: int) -> list[pd.Timestamp]:
    """Last trading day of each month that has at least `lookback` observations."""
    last_of_month = pd.Series(index, index=index).groupby(index.to_period("M")).last()
    return [d for d in last_of_month if index.get_loc(d) + 1 >= lookback]


def backtest(
    prices: pd.DataFrame, strategy: Strategy, lookback: int = 252, cost_bps: float = 10.0
) -> BacktestResult:
    """Monthly-rebalanced backtest with proportional transaction costs.

    Weights decided at the close of day t use returns up to t and earn returns from t+1.
    Holdings drift between rebalances; costs = turnover * cost_bps / 10_000.
    """
    returns = prices.dropna().pct_change().dropna()
    columns = list(returns.columns)
    rebalances = set(rebalance_dates(returns.index, lookback))

    held = np.zeros(len(columns))
    pending: np.ndarray | None = None
    net: dict[pd.Timestamp, float] = {}
    targets: dict[pd.Timestamp, dict[str, float]] = {}
    total_turnover = 0.0

    for i, day in enumerate(returns.index):
        cost = 0.0
        if pending is not None:
            turnover = float(np.abs(pending - held).sum())
            cost = turnover * cost_bps / 10_000
            total_turnover += turnover
            held, pending = pending, None
        if held.sum() > 0:
            r = returns.iloc[i].to_numpy()
            gross = float(held @ r)
            net[day] = gross - cost
            held = held * (1 + r) / (1 + gross)
        if day in rebalances:
            window = returns.iloc[i + 1 - lookback : i + 1]
            weights = strategy(window)
            targets[day] = weights
            pending = np.array([weights.get(c, 0.0) for c in columns])

    return BacktestResult(
        returns=pd.Series(net, dtype=float),
        weights=pd.DataFrame(targets).T.reindex(columns=columns).fillna(0.0),
        turnover=total_turnover,
    )


def performance_summary(returns: pd.Series) -> dict[str, float]:
    """Total return, CAGR, annualized volatility, Sharpe (rf=0) and max drawdown."""
    total = float((1 + returns).prod() - 1)
    years = len(returns) / TRADING_DAYS
    vol = annualized_volatility(returns)
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1,
        "volatility": vol,
        "sharpe": _sharpe(annualized_return(returns), vol),
        "max_drawdown": max_drawdown(returns),
    }


def compare_strategies(
    prices: pd.DataFrame,
    strategies: dict[str, Strategy],
    benchmark: pd.Series | None = None,
    lookback: int = 252,
    cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every strategy (and an optional benchmark) over the same period.

    Returns (summary metrics table, cumulative growth-of-1 curves).
    """
    results = {n: backtest(prices, s, lookback, cost_bps).returns for n, s in strategies.items()}
    start = max(r.index[0] for r in results.values())
    series = {n: r.loc[start:] for n, r in results.items()}
    if benchmark is not None:
        bench = benchmark.pct_change().dropna()
        series["Benchmark"] = bench.loc[start : max(r.index[-1] for r in results.values())]
    summary = pd.DataFrame({n: performance_summary(r) for n, r in series.items()}).T
    curves = pd.DataFrame({n: (1 + r).cumprod() for n, r in series.items()})
    return summary, curves
