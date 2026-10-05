import numpy as np
import pandas as pd
import pytest

from src.optimization import max_sharpe_weights, min_variance_weights, risk_parity_weights


def _cov(vols, corr=None):
    vols = np.array(vols)
    corr = np.eye(len(vols)) if corr is None else np.array(corr)
    names = [chr(65 + i) for i in range(len(vols))]
    return pd.DataFrame(np.outer(vols, vols) * corr, index=names, columns=names)


def test_min_variance_uncorrelated_two_assets():
    # w_A = var_B / (var_A + var_B) = 0.04 / 0.05 = 0.8
    w = min_variance_weights(_cov([0.1, 0.2]))
    assert w["A"] == pytest.approx(0.8, abs=1e-4)
    assert w["B"] == pytest.approx(0.2, abs=1e-4)


def test_max_sharpe_uncorrelated_equal_vol():
    # tangency weights proportional to mu / var -> [0.1, 0.2] -> 1/3, 2/3
    w = max_sharpe_weights(pd.Series([0.1, 0.2], index=["A", "B"]), _cov([0.2, 0.2]))
    assert w["A"] == pytest.approx(1 / 3, abs=1e-3)
    assert w["B"] == pytest.approx(2 / 3, abs=1e-3)


def test_risk_parity_uncorrelated_is_inverse_vol():
    w = risk_parity_weights(_cov([0.1, 0.2]))
    assert w["A"] == pytest.approx(2 / 3, abs=1e-3)
    assert w["B"] == pytest.approx(1 / 3, abs=1e-3)


@pytest.mark.parametrize("fn", [min_variance_weights, risk_parity_weights])
def test_weights_are_valid_and_respect_cap(fn):
    cov = _cov([0.1, 0.25, 0.3], [[1, 0.2, 0.1], [0.2, 1, 0.3], [0.1, 0.3, 1]])
    w = fn(cov, max_weight=0.5)
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(0 <= x <= 0.5 + 1e-9 for x in w.values())


def test_infeasible_cap_raises():
    with pytest.raises(ValueError):
        min_variance_weights(_cov([0.1, 0.2, 0.3]), max_weight=0.3)
