"""Portfolio weight optimizers (long-only, fully invested). Inputs are annualized."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _solve(objective, n: int, max_weight: float, x0: np.ndarray | None = None) -> np.ndarray:
    """Minimize `objective` over weights with sum=1 and 0 <= w <= max_weight."""
    if max_weight * n < 1 - 1e-9:
        raise ValueError(f"max_weight={max_weight} is infeasible for {n} assets")
    start = np.full(n, 1 / n) if x0 is None else x0
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    return np.clip(result.x, 0.0, None) / np.clip(result.x, 0.0, None).sum()


def _as_weights(index: pd.Index, w: np.ndarray) -> dict[str, float]:
    return {str(t): float(x) for t, x in zip(index, w, strict=True)}


def min_variance_weights(cov: pd.DataFrame, max_weight: float = 1.0) -> dict[str, float]:
    """Long-only minimum-variance portfolio."""
    sigma = cov.to_numpy()
    w = _solve(lambda w: w @ sigma @ w, len(cov), max_weight)
    return _as_weights(cov.index, w)


def max_sharpe_weights(
    mu: pd.Series, cov: pd.DataFrame, max_weight: float = 1.0, risk_free: float = 0.0
) -> dict[str, float]:
    """Long-only maximum Sharpe (tangency) portfolio from annualized returns and covariance."""
    m, sigma = mu.to_numpy(), cov.to_numpy()

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(w @ sigma @ w)
        return -(w @ m - risk_free) / vol if vol > 0 else 0.0

    w = _solve(neg_sharpe, len(mu), max_weight)
    return _as_weights(cov.index, w)


def risk_parity_weights(cov: pd.DataFrame, max_weight: float = 1.0) -> dict[str, float]:
    """Equal risk contribution portfolio (each asset contributes the same share of variance)."""
    sigma = cov.to_numpy()
    n = len(cov)

    def dispersion(w: np.ndarray) -> float:
        contributions = w * (sigma @ w)
        return float(((contributions - contributions.mean()) ** 2).sum())

    w = _solve(dispersion, n, max_weight)
    return _as_weights(cov.index, w)
