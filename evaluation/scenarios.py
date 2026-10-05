"""Evaluation scenarios: the report pipeline must stay honest in every situation."""

from dataclasses import dataclass, field
from datetime import datetime

from evaluation.fake_market import patched_market
from src.guardrails import Check, check_allocation
from src.pipeline import finalize_report
from src.tools import TOOL_OUTPUTS

NOW = datetime(2026, 10, 5, 12, 0)
NEUTRAL_NARRATIVE = (
    "### Résumé exécutif\nPortefeuille analysé à partir des données disponibles.\n\n"
    "### Lecture des actions\nForces et faiblesses décrites en mots.\n\n"
    "### Stratégie d'investissement\nHorizon moyen terme, révisions régulières.\n\n"
    "### Risques à surveiller\nRisque de marché et risque sectoriel.\n\n"
    "### Conclusion\nRecommandation prudente, donnée indisponible signalée.\n"
)
MISSING_MARKERS = {
    "risk": "Métriques de risque : donnée indisponible",
    "allocation": "Allocation et cash restant : donnée indisponible",
    "backtest": "Backtest : donnée indisponible",
}


@dataclass(frozen=True)
class Scenario:
    """Inputs of a run and what the report is expected to look like."""

    name: str
    tickers: list[str]
    budget: float = 5000.0
    profile: str = "moderate"
    invalid: frozenset[str] = frozenset()
    currencies: dict[str, str] = field(default_factory=dict)
    history_days: int = 800
    narrative: str = NEUTRAL_NARRATIVE
    expected_missing: frozenset[str] = frozenset()
    expect_valid: bool = True
    expect_zero_investment: bool = False


@dataclass(frozen=True)
class ScenarioResult:
    """Outcome of every check for one scenario."""

    name: str
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


SCENARIOS = [
    Scenario("nominal", ["AAPL", "MSFT", "ASML.AS"], currencies={"AAPL": "USD", "MSFT": "USD"}),
    Scenario("one_invalid_ticker", ["AAPL", "XXXX", "MSFT"], invalid=frozenset({"XXXX"})),
    Scenario(
        "all_tickers_invalid",
        ["XXXX", "YYYY"],
        invalid=frozenset({"XXXX", "YYYY"}),
        expected_missing=frozenset({"risk", "allocation", "backtest"}),
    ),
    Scenario("single_ticker", ["AAPL"]),
    Scenario("budget_below_one_share", ["AAPL", "MSFT"], budget=1.0, expect_zero_investment=True),
    Scenario("zero_budget", ["AAPL", "MSFT"], budget=0.0, expect_zero_investment=True),
    Scenario("very_large_budget", ["AAPL", "MSFT"], budget=1_000_000_000.0),
    Scenario(
        "short_history",
        ["AAPL", "MSFT"],
        history_days=100,
        expected_missing=frozenset({"backtest"}),
    ),
    Scenario(
        "mixed_currencies",
        ["AAPL", "VOD.L", "BNP.PA"],
        currencies={"AAPL": "USD", "VOD.L": "GBp"},
    ),
    Scenario(
        "llm_invents_a_return",
        ["AAPL", "MSFT"],
        narrative=NEUTRAL_NARRATIVE + "\nRendement attendu de 12,5 % par an.\n",
        expect_valid=False,
    ),
    Scenario(
        "llm_invents_an_fx_rate",
        ["AAPL", "MSFT"],
        currencies={"AAPL": "USD", "MSFT": "USD"},
        narrative=NEUTRAL_NARRATIVE + "\nConversion à 1 USD = 0,92 EUR.\n",
        expect_valid=False,
    ),
    Scenario(
        "llm_forgets_the_conclusion",
        ["AAPL", "MSFT"],
        narrative="### Résumé exécutif\nPortefeuille concentré.\n",
        expect_valid=False,
    ),
]


def run_scenario(scenario: Scenario) -> ScenarioResult:
    """Run the full report pipeline offline and score it against the scenario's expectations."""
    TOOL_OUTPUTS.clear()
    try:
        with patched_market(scenario.invalid, scenario.currencies, scenario.history_days):
            report, validation = finalize_report(
                scenario.narrative, scenario.tickers, scenario.budget, scenario.profile, NOW
            )
    except Exception as exc:  # noqa: BLE001 - evaluation boundary: any crash is a failed scenario
        return ScenarioResult(scenario.name, [Check("no_exception", False, repr(exc))])

    failed = [c.name for c in validation.checks if not c.passed]
    checks = [
        Check("no_exception", True),
        Check(
            "guardrails_verdict",
            validation.passed == scenario.expect_valid,
            f"expected {'valid' if scenario.expect_valid else 'invalid'}, failed checks: {failed}",
        ),
    ]
    for block, marker in MISSING_MARKERS.items():
        expected = block in scenario.expected_missing
        checks.append(
            Check(
                f"{block}_missing_flag",
                (marker in report) == expected,
                f"expected {'flagged' if expected else 'present'}",
            )
        )
    if "allocation" not in scenario.expected_missing:
        budget_check = check_allocation(TOOL_OUTPUTS, scenario.budget)
        checks.append(Check("allocation_within_budget", budget_check.passed, budget_check.detail))
        if scenario.expect_zero_investment:
            invested = f"Total investi** : {0:.2f}".replace(".", ",")
            checks.append(Check("nothing_invested", invested in report, "expected 0 invested"))
    return ScenarioResult(scenario.name, checks)


def run_all() -> list[ScenarioResult]:
    return [run_scenario(s) for s in SCENARIOS]
