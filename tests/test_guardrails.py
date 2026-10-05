import json

import pytest

from src.guardrails import (
    check_allocation,
    check_missing_data_acknowledged,
    check_numbers_traceable,
    check_sections,
    extract_numbers,
    format_validation,
    validate_report,
)

ALLOCATION = json.dumps(
    {
        "total_budget": 5000.0,
        "total_invested": 4780.16,
        "cash_remaining": 219.84,
        "allocations": {
            "AAPL": {
                "recommended_weight": "57.7%",
                "allocated_amount": "2885.0",
                "current_price": "333.52",
                "shares_to_buy": 8,
                "real_amount_invested": "2668.16",
            },
            "MSFT": {
                "recommended_weight": "42.3%",
                "allocated_amount": "2115.0",
                "current_price": "528.0",
                "shares_to_buy": 4,
                "real_amount_invested": "2112.0",
            },
        },
    }
)
BACKTEST = json.dumps(
    {"strategies": {"Equal weight": {"cagr": "20.38%", "max_drawdown": "-26.6%"}}}
)
REPORT = """# Rapport
## 5. ALLOCATION
| AAPL | 57,7 % | 2 885,00 € | 333,52 | 8 | 2 668,16 € |
| MSFT | 42,3 % | 2 115,00 € | 528,00 | 4 | 2 112,00 € |
Total investi : 4 780,16 € — **CASH RESTANT** : 219,84 € sur 5 000 €.
## 6. RISQUES
## 7. BACKTEST
Equal weight : CAGR 20,38 %, drawdown –26,6 % (période 2024-11-01).
## 10. CONCLUSION
"""
OUTPUTS = [ALLOCATION, BACKTEST]


def test_extract_numbers_handles_french_formats():
    nums = dict(extract_numbers("Montant 2\u00a0885,00 € et drawdown \u201326,6 %"))
    assert nums[2885.0] == 2 and nums[26.6] == 1


def test_extract_financial_only_ignores_plain_integers_and_years():
    text = "7. BACKTEST sur 252 jours en 2024, 10 bps, 8 actions, 57,7 %, 100 %"
    values = [v for v, _ in extract_numbers(text, financial_only=True)]
    assert values == [57.7, 100.0]


def test_valid_report_passes_every_check():
    result = validate_report(REPORT, OUTPUTS, budget=5000.0)
    assert result.passed, [c for c in result.checks if not c.passed]


def test_invented_number_is_flagged():
    check = check_numbers_traceable(REPORT + "\nRendement attendu 12,34 %.", OUTPUTS, 5000.0)
    assert not check.passed and "12.34" in check.detail


def test_rounding_to_report_precision_is_accepted_but_not_other_values():
    assert check_numbers_traceable("CAGR 20,4 %", OUTPUTS, 5000.0).passed
    assert not check_numbers_traceable("CAGR 20,5 %", OUTPUTS, 5000.0).passed


def test_missing_section_is_flagged():
    check = check_sections(REPORT.replace("BACKTEST", "TESTS"))
    assert not check.passed and "backtest" in check.detail


def test_fallback_must_be_acknowledged():
    outputs = ["⚠️ DATA UNAVAILABLE — Yahoo"]
    assert not check_missing_data_acknowledged("tout va bien", outputs).passed
    assert check_missing_data_acknowledged("donnée indisponible", outputs).passed


def test_overspent_allocation_is_flagged():
    bad = json.loads(ALLOCATION)
    bad["total_invested"] = 5200.0
    assert not check_allocation([json.dumps(bad)], 5000.0).passed
    assert check_allocation([ALLOCATION], 5000.0).passed


def test_weights_not_summing_to_100_are_flagged():
    bad = json.loads(ALLOCATION)
    bad["allocations"]["MSFT"]["recommended_weight"] = "30.0%"
    assert not check_allocation([json.dumps(bad)], 5000.0).passed


def test_format_validation_reports_status():
    ok = format_validation(validate_report(REPORT, OUTPUTS, 5000.0))
    assert "VALIDÉ" in ok and "NON VALIDÉ" not in ok
    ko = format_validation(validate_report(REPORT + " 9,99 %", OUTPUTS, 5000.0))
    assert "NON VALIDÉ" in ko and "numbers_traceable : ÉCHEC" in ko


@pytest.mark.parametrize("text", ["", "aucun chiffre ici"])
def test_report_without_numbers_has_nothing_untraceable(text):
    assert check_numbers_traceable(text, OUTPUTS, 5000.0).passed
