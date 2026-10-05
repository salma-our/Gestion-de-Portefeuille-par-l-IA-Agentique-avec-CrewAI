"""Pure financial calculations. All numbers shown in reports come from here, never from the LLM."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns (fraction, e.g. 0.25 = 25%)."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def annualized_return(returns: pd.Series) -> float:
    """Arithmetic annualized mean of daily returns (fraction)."""
    return float(returns.mean() * TRADING_DAYS)


def sharpe_ratio(ann_return: float, ann_vol: float) -> float:
    """Return / volatility (risk-free rate assumed 0); 0.0 if volatility is zero."""
    return ann_return / ann_vol if ann_vol else 0.0


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss of the cumulative return curve (negative fraction)."""
    cumulative = (1 + returns).cumprod()
    return float(((cumulative - cumulative.cummax()) / cumulative.cummax()).min())


def inverse_volatility_weights(volatilities: dict[str, float]) -> dict[str, float]:
    """Risk-parity style weights proportional to 1/volatility; ignores non-positive vols."""
    inv = {t: 1 / v for t, v in volatilities.items() if v > 0}
    total = sum(inv.values())
    return {t: x / total for t, x in inv.items()} if total else {}


@dataclass(frozen=True)
class Position:
    """Whole-share position for one ticker."""

    weight: float
    target_amount: float
    price: float
    shares: int
    invested: float


def allocate(
    weights: dict[str, float], prices: dict[str, float], budget: float
) -> tuple[dict[str, Position], float]:
    """Buy whole shares per weight; return positions and leftover cash."""
    positions: dict[str, Position] = {}
    for ticker, weight in weights.items():
        price = prices.get(ticker, 0.0)
        target = budget * weight
        shares = int(target // price) if price > 0 else 0
        positions[ticker] = Position(weight, target, price, shares, shares * price)
    cash = budget - sum(p.invested for p in positions.values())
    return positions, cash
