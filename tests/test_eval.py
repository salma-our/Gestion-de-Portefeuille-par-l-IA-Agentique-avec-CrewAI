import logging

import pytest

from evaluation.scenarios import SCENARIOS, Scenario, run_scenario


@pytest.fixture(autouse=True)
def quiet_tool_errors():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario_passes(scenario):
    result = run_scenario(scenario)
    assert result.passed, [(c.name, c.detail) for c in result.checks if not c.passed]


def test_scenario_names_are_unique():
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names))


def test_checks_can_fail_on_wrong_expectations():
    wrong_verdict = run_scenario(Scenario("x", ["AAPL", "MSFT"], expect_valid=False))
    wrong_missing = run_scenario(
        Scenario("y", ["AAPL", "MSFT"], expected_missing=frozenset({"backtest"}))
    )
    assert not wrong_verdict.passed and not wrong_missing.passed
