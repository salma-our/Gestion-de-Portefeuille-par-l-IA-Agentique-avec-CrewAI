"""Rule-based report validation: no LLM involved, every check is deterministic."""

import json
import re
from dataclasses import dataclass, field

FALLBACK_MARKER = "DATA UNAVAILABLE"
REQUIRED_SECTIONS = ("allocation", "backtest", "cash", "risque", "conclusion")
ALWAYS_ALLOWED = (100.0,)

_NUMBER = re.compile(r"(?<![\w.,])-?\d{1,3}(?: \d{3})+(?:[.,]\d+)?|(?<![\w.,])-?\d+(?:[.,]\d+)?")
_UNIT_AFTER = re.compile(r"\s?(?:%|€|EUR\b|USD\b)")


@dataclass(frozen=True)
class Check:
    """Outcome of one validation rule."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class GuardrailResult:
    """All checks for one report."""

    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _normalize(text: str) -> str:
    for ch in ("\u00a0", "\u202f", "\u2009"):
        text = text.replace(ch, " ")
    return text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2011", "-")


def extract_numbers(text: str, financial_only: bool = False) -> list[tuple[float, int]]:
    """Return (absolute value, decimals) for each number; optionally only financial-looking ones.

    Financial-looking means it has decimals or a %, EUR, USD unit; plain integers (section
    numbers, years, window sizes) are ignored.
    """
    text = _normalize(text)
    found = []
    for m in _NUMBER.finditer(text):
        raw = m.group().replace(" ", "").replace(",", ".")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        has_unit = (
            bool(_UNIT_AFTER.match(text, m.end())) or text[max(0, m.start() - 1) : m.start()] == "$"
        )
        if financial_only and decimals == 0 and not has_unit:
            continue
        found.append((abs(float(raw)), decimals))
    return found


def _matches(value: float, decimals: int, truth: list[float]) -> bool:
    tolerance = 0.5 * 10**-decimals + 1e-9
    return any(abs(value - t) <= tolerance for t in truth)


def check_sections(report: str) -> Check:
    lowered = report.lower()
    missing = [s for s in REQUIRED_SECTIONS if s not in lowered]
    return Check(
        "required_sections", not missing, f"missing: {', '.join(missing)}" if missing else ""
    )


def check_numbers_traceable(report: str, tool_outputs: list[str], budget: float) -> Check:
    """Every financial-looking number in the report must come from a tool output (or the budget)."""
    truth = [v for out in tool_outputs for v, _ in extract_numbers(out)]
    truth += [budget, *ALWAYS_ALLOWED]
    untraceable = []
    for value, decimals in extract_numbers(report, financial_only=True):
        if not _matches(value, decimals, truth):
            untraceable.append(f"{value:.{decimals}f}")
    unique = list(dict.fromkeys(untraceable))
    detail = f"{len(unique)} untraceable: {', '.join(unique[:10])}" if unique else ""
    return Check("numbers_traceable", not unique, detail)


def check_missing_data_acknowledged(report: str, tool_outputs: list[str]) -> Check:
    """If a tool returned the fallback message, the report must say 'indisponible'."""
    failed = any(FALLBACK_MARKER in out for out in tool_outputs)
    ok = not failed or "indisponible" in report.lower()
    return Check(
        "missing_data_acknowledged", ok, "tool data missing but not flagged" if not ok else ""
    )


def check_allocation(tool_outputs: list[str], budget: float) -> Check:
    """Allocation tool output must respect the budget and be internally consistent."""
    for out in tool_outputs:
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "allocations" not in data:
            continue
        problems = []
        if data["total_invested"] > budget + 0.01:
            problems.append("invested exceeds budget")
        if data["cash_remaining"] < -0.01:
            problems.append("negative cash")
        weights = [float(a["recommended_weight"].rstrip("%")) for a in data["allocations"].values()]
        if abs(sum(weights) - 100) > 0.1 * len(weights):
            problems.append(f"weights sum to {sum(weights):.1f}%")
        for t, a in data["allocations"].items():
            expected = a["shares_to_buy"] * float(a["current_price"])
            if abs(expected - float(a["real_amount_invested"])) > 0.05:
                problems.append(f"{t}: shares x price mismatch")
        return Check("allocation_consistent", not problems, "; ".join(problems))
    return Check("allocation_consistent", True, "no allocation output to check")


def validate_report(report: str, tool_outputs: list[str], budget: float) -> GuardrailResult:
    """Run every guardrail on a generated report."""
    return GuardrailResult(
        [
            check_sections(report),
            check_numbers_traceable(report, tool_outputs, budget),
            check_missing_data_acknowledged(report, tool_outputs),
            check_allocation(tool_outputs, budget),
        ]
    )


def format_validation(result: GuardrailResult) -> str:
    """Markdown footer summarising the validation."""
    status = "VALIDÉ" if result.passed else "NON VALIDÉ"
    lines = [f"\n---\n**Validation automatique : {status}**\n"]
    for c in result.checks:
        mark = "OK" if c.passed else "ÉCHEC"
        lines.append(f"- {c.name} : {mark}" + (f" ({c.detail})" if c.detail else ""))
    return "\n".join(lines) + "\n"
