"""Weight strategies: each maps a window of daily returns to {ticker: weight}."""

from collections.abc import Callable

import pandas as pd

from src.finance import TRADING_DAYS, inverse_volatility_weights
from src.optimization import max_sharpe_weights, min_variance_weights, risk_parity_weights

Strategy = Callable[[pd.DataFrame], dict[str, float]]


def equal_weight(window: pd.DataFrame) -> dict[str, float]:
    """1/N benchmark strategy."""
    return {str(t): 1 / window.shape[1] for t in window.columns}


def inverse_volatility(window: pd.DataFrame) -> dict[str, float]:
    """Weights proportional to 1/volatility (the project's original allocation)."""
    return inverse_volatility_weights((window.std() * TRADING_DAYS**0.5).to_dict())


def min_variance(window: pd.DataFrame, max_weight: float = 1.0) -> dict[str, float]:
    """Long-only minimum variance."""
    return min_variance_weights(window.cov() * TRADING_DAYS, max_weight)


def max_sharpe(window: pd.DataFrame, max_weight: float = 1.0) -> dict[str, float]:
    """Long-only maximum Sharpe using the window's mean returns."""
    return max_sharpe_weights(window.mean() * TRADING_DAYS, window.cov() * TRADING_DAYS, max_weight)


def risk_parity(window: pd.DataFrame, max_weight: float = 1.0) -> dict[str, float]:
    """Equal risk contribution."""
    return risk_parity_weights(window.cov() * TRADING_DAYS, max_weight)
