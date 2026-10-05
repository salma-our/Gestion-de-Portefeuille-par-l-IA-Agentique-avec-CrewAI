"""Final report assembly: Python-rendered numbers + LLM narrative + guardrail footer."""

from datetime import datetime

from src.facts import collect_facts
from src.guardrails import GuardrailResult, format_validation, validate_report
from src.report_builder import build_report
from src.tools import TOOL_OUTPUTS


def finalize_report(
    narrative: str, tickers: list[str], budget: float, profile: str, now: datetime | None = None
) -> tuple[str, GuardrailResult]:
    """Build the full Markdown report and validate it; returns (markdown, validation)."""
    facts = collect_facts(tickers, budget, profile)
    report = build_report(facts, narrative, now or datetime.now())
    validation = validate_report(report, TOOL_OUTPUTS, budget, narrative)
    return report + format_validation(validation), validation
