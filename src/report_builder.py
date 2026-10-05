"""Deterministic Markdown rendering of the numeric report sections (the LLM never writes these)."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

MISSING = "donnée indisponible"
PROFILE_LABELS = {"conservative": "Conservateur", "moderate": "Modéré", "aggressive": "Agressif"}


@dataclass(frozen=True)
class Facts:
    """Parsed tool outputs; None means the tool had no usable data."""

    tickers: list[str]
    budget: float
    profile: str
    stocks: dict[str, dict[str, Any] | None]
    risk: dict[str, Any] | None
    allocation: dict[str, Any] | None
    backtest: dict[str, Any] | None


def _text(value: Any) -> str:
    """Display a tool value: decimal comma, spaced percent, spaced thousands, N/A -> missing."""
    if value is None or value == "N/A":
        return MISSING
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}".replace(",", " ")
    return re.sub(r"(\d)\.(\d)", r"\1,\2", str(value)).replace("%", " %")


def _money(value: Any) -> str:
    """Format an amount as '2 885,00 €'."""
    return f"{float(value):,.2f}".replace(",", " ").replace(".", ",") + " €"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]
    return "\n".join(lines)


def render_stocks(facts: Facts) -> str:
    keys = [
        "name",
        "sector",
        "current_price",
        "variation_1d",
        "variation_1m",
        "pe_ratio",
        "beta",
        "dividend_yield",
        "52w_high",
        "52w_low",
    ]
    headers = ["Ticker", "Nom", "Secteur", "Prix", "Var. 1 j", "Var. 1 m", "P/E", "Bêta"]
    headers += ["Dividende (%)", "Plus haut 52 s", "Plus bas 52 s"]
    rows = []
    for ticker in facts.tickers:
        data = facts.stocks.get(ticker)
        rows.append([ticker, *(_text(data.get(k) if data else None) for k in keys)])
    return "## 1. Données par action\n\n" + _table(headers, rows)


def render_risk(facts: Facts) -> str:
    out = "## 2. Analyse du risque\n\n"
    if not facts.risk:
        return out + f"Métriques de risque : {MISSING}."
    metrics = facts.risk.get("metrics_per_asset", {})
    rows = [
        [
            t,
            _text(m.get("annual_volatility")),
            _text(m.get("annual_return_estimate")),
            _text(m.get("sharpe_ratio")),
            _text(m.get("max_drawdown")),
        ]
        for t, m in metrics.items()
    ]
    headers = ["Ticker", "Volatilité annuelle", "Rendement annuel estimé", "Sharpe", "Drawdown max"]
    out += _table(headers, rows)
    corr = facts.risk.get("correlation_matrix")
    if isinstance(corr, dict) and corr:
        names = list(corr)
        crows = [[a, *(_text(corr[a].get(b)) for b in names)] for a in names]
        out += "\n\n**Corrélations**\n\n" + _table([""] + names, crows)
    else:
        out += "\n\nCorrélations : non applicables (un seul actif)."
    return out


def render_allocation(facts: Facts) -> str:
    out = "## 3. Allocation recommandée\n\n"
    alloc = facts.allocation
    if not alloc:
        return out + f"Allocation et cash restant : {MISSING}."
    headers = ["Ticker", "Devise", "Prix local", "Taux vers EUR", "Prix (EUR)", "Poids"]
    headers += ["Montant alloué", "Actions", "Investi"]
    rows = [
        [
            t,
            a["currency"],
            _text(a["price_local"]),
            _text(a["fx_rate_to_eur"]),
            _text(a["current_price"]),
            _text(a["recommended_weight"]),
            _money(a["allocated_amount"]),
            _text(a["shares_to_buy"]),
            _money(a["real_amount_invested"]),
        ]
        for t, a in alloc["allocations"].items()
    ]
    out += _table(headers, rows)
    out += f"\n\n- **Budget** : {_money(alloc['total_budget'])}"
    out += f"\n- **Total investi** : {_money(alloc['total_invested'])}"
    out += f"\n- **Cash restant** : {_money(alloc['cash_remaining'])}"
    out += f"\n- **Méthode** : {alloc['method']}"
    return out


def render_backtest(facts: Facts) -> str:
    out = "## 4. Backtest\n\n"
    bt = facts.backtest
    if not bt:
        return out + f"Backtest : {MISSING}."
    rows = [
        [
            name,
            _text(m["total_return"]),
            _text(m["cagr"]),
            _text(m["volatility"]),
            _text(m["sharpe"]),
            _text(m["max_drawdown"]),
        ]
        for name, m in bt["strategies"].items()
    ]
    headers = ["Stratégie", "Rendement total", "CAGR", "Volatilité", "Sharpe", "Drawdown max"]
    out += _table(headers, rows)
    out += f"\n\n- **Période** : {bt['period']}\n- **Hypothèses** : {bt['assumptions']}"
    if bt.get("benchmark") == MISSING:
        out += f"\n- **Benchmark** : {MISSING}"
    return out


def build_report(facts: Facts, narrative: str, now: datetime) -> str:
    """Assemble numeric sections (from tools) and the LLM's qualitative narrative."""
    profile = PROFILE_LABELS.get(facts.profile, facts.profile)
    header = (
        "# Rapport PortfolioPilot\n\n"
        f"*Généré le {now:%d/%m/%Y %H:%M} · Profil : {profile} · Budget : {_money(facts.budget)} · "
        f"Titres : {', '.join(facts.tickers)} · Données : Yahoo Finance (yfinance)*"
    )
    commentary = narrative.strip() or f"Commentaire : {MISSING}."
    sections = [
        header,
        render_stocks(facts),
        render_risk(facts),
        render_allocation(facts),
        render_backtest(facts),
        "## 5. Analyse et recommandations\n\n" + commentary,
        "*Projet éducatif. Ne constitue pas un conseil en investissement.*",
    ]
    return "\n\n".join(sections) + "\n"
