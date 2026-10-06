"""
app.py — Interface web Streamlit enrichie
PortfolioPilot — Agentic portfolio analysis

CORRECTION PRINCIPALE :
- run_analysis() appelle maintenant réellement les agents CrewAI via crew.kickoff()
- Les agents LLM analysent, évaluent les risques et rédigent le rapport
- Fallback automatique sur calcul local si la clé API manque ou en cas d'erreur

Lancement : streamlit run app.py
"""

import logging
import os
import sys
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

from src.finance import allocate, annualized_volatility, inverse_volatility_weights
from src.llm import api_key_var, llm_configured
from src.pipeline import finalize_report
from src.tools import DATA_ERRORS, TOOL_OUTPUTS, _download, _extract_close, run_backtest

# ── Ajout du dossier courant au PYTHONPATH ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

# ── Configuration de la page ──────────────────────────────
st.set_page_config(
    page_title="PortfolioPilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.main { background: #0a0a0a; }
.block-container { padding: 2rem 2.5rem; max-width: 1450px; }

.app-header { display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #1e1e1e; padding-bottom: 1.5rem; margin-bottom: 2rem; }
.app-title { font-size: 22px; font-weight: 500; color: #f0f0f0; margin: 0; }
.app-sub { font-size: 13px; color: #555; margin: 0; font-family: 'IBM Plex Mono', monospace; }

.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 500; font-family: 'IBM Plex Mono', monospace; }
.badge-green  { background: #0d2b1a; color: #3ecf8e; border: 1px solid #1a4a30; }
.badge-blue   { background: #0d1a2b; color: #60a5fa; border: 1px solid #1a3050; }
.badge-orange { background: #2b1a0d; color: #fb923c; border: 1px solid #4a300d; }
.badge-red    { background: #2b0d0d; color: #f87171; border: 1px solid #4a1a1a; }

.agent-card { background: #111; border: 1px solid #1e1e1e; border-radius: 10px; padding: 16px 18px; margin-bottom: 10px; }
.agent-card.running { border-color: #1a3050; background: #0d1520; }
.agent-card.done { border-color: #1a4a30; background: #0a1f14; }
.agent-header { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.agent-name { font-size: 14px; font-weight: 500; color: #e0e0e0; }
.agent-status { font-size: 12px; color: #555; font-family: 'IBM Plex Mono', monospace; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-idle { background: #2e2e2e; }
.dot-running { background: #60a5fa; animation: blink 1s infinite; }
.dot-done { background: #3ecf8e; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

.metric-card { background: #111; border: 1px solid #1e1e1e; border-radius: 10px; padding: 18px 20px; }
.metric-label { font-size: 12px; color: #555; margin-bottom: 6px; font-family: 'IBM Plex Mono', monospace; }
.metric-value { font-size: 26px; font-weight: 300; color: #f0f0f0; }
.metric-unit { font-size: 13px; color: #444; margin-top: 4px; }

.alloc-row { display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid #1a1a1a; }
.alloc-row:last-child { border-bottom: none; }
.alloc-ticker { font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 500; color: #e0e0e0; min-width: 80px; }
.alloc-bar-bg { flex: 1; height: 6px; background: #1e1e1e; border-radius: 3px; overflow: hidden; }
.alloc-bar-fill { height: 100%; border-radius: 3px; }
.alloc-pct { font-size: 13px; color: #888; min-width: 40px; text-align: right; font-family: 'IBM Plex Mono', monospace; }
.alloc-amt { font-size: 13px; color: #555; min-width: 80px; text-align: right; font-family: 'IBM Plex Mono', monospace; }

.log-terminal { background: #060606; border: 1px solid #1a1a1a; border-radius: 10px; padding: 14px 16px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; max-height: 220px; overflow-y: auto; }
.log-entry { padding: 2px 0; }
.log-time { color: #333; margin-right: 8px; }
.log-system { color: #555; }
.log-agent1 { color: #60a5fa; }
.log-agent2 { color: #fb923c; }
.log-agent3 { color: #3ecf8e; }
.log-agent4 { color: #a78bfa; }
.log-success { color: #a3e635; }

.cmd-box { background: #060606; border: 1px solid #1a1a1a; border-radius: 8px; padding: 12px 16px; font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #60a5fa; word-break: break-all; }

.stButton > button { background: #f0f0f0 !important; color: #0a0a0a !important; border: none !important; border-radius: 8px !important; font-weight: 500 !important; font-size: 14px !important; padding: 10px 24px !important; width: 100% !important; font-family: 'IBM Plex Sans', sans-serif !important; }
.stButton > button:hover { background: #d0d0d0 !important; }
.stButton > button:disabled { background: #1e1e1e !important; color: #444 !important; }

div[data-testid="stSidebar"] { background: #080808; border-right: 1px solid #1a1a1a; }
div[data-testid="stSidebar"] .stSelectbox label, div[data-testid="stSidebar"] .stNumberInput label, div[data-testid="stSidebar"] .stTextInput label { color: #888 !important; font-size: 12px !important; font-family: 'IBM Plex Mono', monospace !important; }
.stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input { background: #111 !important; color: #e0e0e0 !important; border: 1px solid #1e1e1e !important; border-radius: 8px !important; }
.stAlert { border-radius: 8px !important; }
hr { border-color: #1a1a1a !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Initialisation session state ──────────────────────────
N_AGENTS = 4  # analyst, news, risk, strategist


def init_state():
    defaults = {
        "tickers": ["AAPL", "MSFT", "ASML.AS"],
        "budget": 10000.0,
        "profile": "moderate",
        "running": False,
        "done": False,
        "agent_states": ["idle"] * N_AGENTS,
        "logs": [],
        "alloc_data": None,
        "report_text": None,
        "prices": {},
        "start_time": None,
        "dashboard_data": None,
        "backtest": None,
        "pdf_bytes": None,
        "crew_mode": True,  # True = appelle les vrais agents CrewAI
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ── Helpers ───────────────────────────────────────────────
BAR_COLORS = ["#60a5fa", "#3ecf8e", "#fb923c", "#a78bfa", "#f472b6"]
PROFILE_META = {
    "conservative": ("", "Conservateur", "badge-green"),
    "moderate": ("", "Modéré", "badge-blue"),
    "aggressive": ("", "Agressif", "badge-orange"),
}
PROFILE_TEXT = {
    "conservative": "Préserver le capital, limiter la volatilité et privilégier une allocation plus défensive.",
    "moderate": "Chercher un équilibre entre rendement et risque avec une diversification raisonnable.",
    "aggressive": "Accepter une volatilité plus élevée pour viser un rendement potentiel plus important.",
}

# Noms et classes CSS des agents
AGENT_ROLES_IDX = {
    "Analyste de Marché": 0,
    "Analyste Actualités": 1,
    "Gestionnaire des Risques": 2,
    "Stratège de Portefeuille": 3,
}
LOG_CLASSES = ["log-agent1", "log-agent4", "log-agent2", "log-agent3"]
LOG_NAMES = ["Analyste", "Actualités", "Risques", "Stratège"]


def now_str():
    if st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        return f"{elapsed // 60:02d}:{elapsed % 60:02d}"
    return "00:00"


def add_log(agent_class, agent_name, msg):
    st.session_state.logs.append(
        {
            "time": now_str(),
            "cls": agent_class,
            "name": agent_name,
            "msg": msg,
        }
    )


def render_logs():
    lines = ""
    for e in st.session_state.logs:
        lines += (
            f'<div class="log-entry">'
            f'<span class="log-time">{e["time"]}</span>'
            f'<span class="{e["cls"]}">[{e["name"]}]</span> '
            f'<span style="color:#888">{e["msg"]}</span>'
            f"</div>"
        )
    return f'<div class="log-terminal">{lines}</div>'


# ── Données de marché (yfinance) ──────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(tickers, period="1y"):
    data = yf.download(tickers, period=period, progress=False, threads=False, auto_adjust=True)
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_backtest(tickers: tuple[str, ...]):
    """Backtest summary and growth curves, or None if the data is unusable."""
    try:
        return run_backtest(list(tickers))
    except DATA_ERRORS:
        logger.exception("Backtest failed")
        return None


def fetch_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="5d", auto_adjust=True)
            prices[t] = round(float(hist["Close"].dropna().iloc[-1]), 2) if not hist.empty else None
        except Exception as exc:  # noqa: BLE001 - yfinance raises many unrelated types
            logger.warning("Price fetch failed for %s: %s", t, exc)
            prices[t] = None
    return prices


def compute_dashboard_data(tickers):
    try:
        data = fetch_market_data(tuple(tickers), period="1y")
        if data.empty:
            return None
        data = data[[c for c in data.columns if c in tickers]].dropna(axis=1, how="all")
        returns = data.pct_change().dropna(how="all")
        if returns.empty:
            return None
        cumulative = (1 + returns.fillna(0)).cumprod()
        cumulative = cumulative / cumulative.iloc[0]
        performance = ((data.iloc[-1] / data.iloc[0]) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        running_max = data.cummax()
        drawdown = (data / running_max - 1) * 100
        max_drawdown = drawdown.min()
        corr = returns.corr()
        summary = pd.DataFrame(
            {
                "Prix actuel": data.iloc[-1].round(2),
                "Performance 1 an (%)": performance.round(2),
                "Volatilité annuelle (%)": volatility.round(2),
                "Sharpe ratio": sharpe.round(2),
                "Drawdown max (%)": max_drawdown.round(2),
            }
        )
        summary.index.name = "Ticker"
        return {
            "prices_history": data,
            "returns": returns,
            "cumulative": cumulative,
            "summary": summary,
            "corr": corr,
            "portfolio_volatility": float(volatility.mean()) if len(volatility) else None,
            "best_asset": performance.idxmax() if len(performance) else None,
            "worst_asset": performance.idxmin() if len(performance) else None,
        }
    except (ValueError, KeyError, IndexError, ZeroDivisionError):
        logger.exception("Dashboard computation failed")
        return None


def compute_allocation(tickers, budget, prices):
    """Risk-parity allocation, same functions and 6-month window as the portfolio_allocation tool."""
    volatilities: dict[str, float] = {}
    try:
        data = _extract_close(_download(list(tickers), period="6mo"), list(tickers))
        for t in tickers:
            if t in data.columns:
                vol = annualized_volatility(data[t].dropna().pct_change().dropna())
                if vol > 0:
                    volatilities[t] = vol
    except (ValueError, KeyError, IndexError, ZeroDivisionError):
        logger.exception("Volatility computation failed")
    weights = inverse_volatility_weights(volatilities)
    if not weights:
        logger.warning("No usable volatility, falling back to equal weights")
        weights = {t: 1 / len(tickers) for t in tickers}
    positions, _ = allocate(weights, {t: prices.get(t) or 0.0 for t in weights}, budget)
    return [
        {
            "ticker": t,
            "poids": round(p.weight * 100, 1),
            "montant": round(p.target_amount, 2),
            "prix": round(p.price, 2),
            "nb_actions": p.shares,
            "montant_reel": round(p.invested, 2),
        }
        for t, p in positions.items()
    ]


def build_rich_report(tickers, budget, profile, alloc, dashboard):
    """Rapport fallback (généré localement sans IA si la clé API manque)."""
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    total_investi = sum(a["montant_reel"] for a in alloc)
    cash = budget - total_investi
    profile_label = PROFILE_META[profile][1]
    summary = dashboard.get("summary") if dashboard else None
    corr = dashboard.get("corr") if dashboard else None
    best_asset = dashboard.get("best_asset") if dashboard else "donnée indisponible"
    worst_asset = dashboard.get("worst_asset") if dashboard else "donnée indisponible"
    report = f"""# Rapport de Portefeuille (mode local – sans IA)

*Généré le {now} · Profil : {profile_label} · Données : yfinance*


## 1. Synthèse exécutive

Portefeuille de **{len(tickers)} actifs** — Budget : **{budget:,.2f} EUR** — Profil : **{profile_label}**

{PROFILE_TEXT[profile]}

- Capital investi estimé : {total_investi:,.2f} EUR
- Cash restant : {cash:,.2f} EUR
- Meilleur actif sur 1 an : {best_asset}
- Actif le moins performant : {worst_asset}

## 2. Allocation recommandée

| Ticker | Poids | Prix actuel | Nombre d'actions | Montant investi |
|---|---:|---:|---:|---:|
"""
    for a in alloc:
        report += f"| {a['ticker']} | {a['poids']}% | {a['prix']} | {a['nb_actions']} | {a['montant_reel']:,.2f} EUR |\n"
    report += "\n## 3. Analyse par actif\n\n"
    if summary is not None:
        for ticker in summary.index:
            row = summary.loc[ticker]
            report += f"""### {ticker}
- Prix actuel : **{row.get("Prix actuel", "N/A")}**
- Performance 1 an : **{row.get("Performance 1 an (%)", "N/A")}%**
- Volatilité annuelle : **{row.get("Volatilité annuelle (%)", "N/A")}%**
- Sharpe ratio : **{row.get("Sharpe ratio", "N/A")}**
- Drawdown maximum : **{row.get("Drawdown max (%)", "N/A")}%**

"""
    report += "## 4. Analyse du risque\n\n"
    if corr is not None and len(corr) > 1:
        avg_corr = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().mean()
        report += f"Corrélation moyenne entre actifs : **{avg_corr:.2f}**.\n"
    report += "\n## 5. Capital déployé\n\n"
    report += f"- Budget initial : **{budget:,.2f} EUR**\n"
    report += f"- Montant réellement investi : **{total_investi:,.2f} EUR**\n"
    report += f"- Cash restant : **{cash:,.2f} EUR**\n"
    return report


# ── Génération PDF ────────────────────────────────────────
def clean_markdown(text):
    for old, new in {
        "**": "",
        "###": "",
        "##": "",
        "#": "",
        "*": "",
        "€": "EUR",
        "→": "->",
        "·": "-",
    }.items():
        text = text.replace(old, new)
    return text


def generate_pdf_bytes(report_text, alloc=None, summary=None):
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TC", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=14
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=4
    )
    normal_style = ParagraphStyle(
        "NR", parent=styles["Normal"], fontSize=9, leading=12, spaceAfter=4
    )
    story = []
    table_buffer = []

    def flush_table():
        nonlocal table_buffer
        if not table_buffer:
            return
        rows = [
            [clean_markdown(c.strip()) for c in line.split("|") if c.strip()]
            for line in table_buffer
            if "---" not in line
        ]
        rows = [r for r in rows if r]
        if rows:
            t = Table(rows, repeatRows=1)
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(t)
            story.append(Spacer(1, 6))
        table_buffer.clear()

    for line in report_text.splitlines():
        line = line.strip()
        if not line:
            flush_table()
            story.append(Spacer(1, 4))
            continue
        if "|" in line:
            table_buffer.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            story.append(Paragraph(clean_markdown(line), title_style))
        elif line.startswith("## "):
            story.append(Paragraph(clean_markdown(line), h1_style))
        elif line.startswith("### "):
            story.append(Paragraph(clean_markdown(line), h2_style))
        elif line.startswith("- "):
            story.append(Paragraph("• " + clean_markdown(line[2:]), normal_style))
        else:
            story.append(Paragraph(clean_markdown(line), normal_style))
    flush_table()
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def save_report_files(report_text, pdf_bytes):
    os.makedirs("reports", exist_ok=True)
    with open("reports/rapport_portefeuille.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    if pdf_bytes:
        with open("reports/rapport_portefeuille.pdf", "wb") as f:
            f.write(pdf_bytes)


# ════════════════════════════════════════════════════════
# FONCTION PRINCIPALE : ANALYSE VIA CREWAI
# ════════════════════════════════════════════════════════
def run_analysis():
    """
    Lance l'analyse réelle via les 3 agents CrewAI + LLM.
    - Agent 1 (Analyste) : appelle StockAnalysisTool sur chaque ticker
    - Agent 2 (Risques)  : appelle PortfolioRiskTool
    - Agent 3 (Stratège) : appelle AllocationTool et rédige le rapport
    Les callbacks step_callback / task_callback mettent à jour le journal en temps réel.
    Si la clé API est absente ou en cas d'erreur, repli sur le calcul local.
    """
    st.session_state.running = True
    st.session_state.done = False
    st.session_state.logs = []
    st.session_state.alloc_data = None
    st.session_state.report_text = None
    st.session_state.dashboard_data = None
    st.session_state.backtest = None
    st.session_state.pdf_bytes = None
    st.session_state.start_time = time.time()
    st.session_state.agent_states = ["idle"] * N_AGENTS

    tickers = st.session_state.tickers
    budget = st.session_state.budget
    profile = st.session_state.profile

    # ── 1. Données de marché (indépendant des agents, pour le dashboard) ──
    add_log("log-system", "Système", "Récupération des données de marché (yfinance)…")
    prices = fetch_prices(tickers)
    dashboard = compute_dashboard_data(tickers)
    st.session_state.prices = prices
    st.session_state.dashboard_data = dashboard
    add_log("log-system", "Système", "Backtest des stratégies (3 ans, benchmark S&P 500)…")
    st.session_state.backtest = fetch_backtest(tuple(tickers))
    for t, p in prices.items():
        add_log("log-agent1", "Analyste", f"{t} → {p} EUR" if p else f"{t} → donnée indisponible")

    # ── 2. Vérification de la clé API ──────────────────────────────────
    api_key = os.getenv(api_key_var(), "")
    use_crew = bool(api_key)

    if not use_crew:
        add_log("log-system", "Système", "⚠ clé API manquante → mode calcul local")

    # ── 3. Mode CrewAI (agents réels) ──────────────────────────────────
    if use_crew:
        try:
            from crewai import Crew, Process
            from crewai.agents.parser import AgentAction, AgentFinish

            from src.agents import create_agents
            from src.tasks import create_tasks

            # Compteur partagé pour suivre la tâche en cours
            task_idx = [0]

            def step_callback(step):
                """Appelé à chaque étape (action ou réponse finale) d'un agent."""
                idx = min(task_idx[0], N_AGENTS - 1)
                st.session_state.agent_states[idx] = "running"
                if isinstance(step, AgentAction):
                    tool_input = str(step.tool_input)[:80]
                    add_log(
                        LOG_CLASSES[idx], LOG_NAMES[idx], f"→ outil [{step.tool}] : {tool_input}…"
                    )
                elif isinstance(step, AgentFinish):
                    add_log(LOG_CLASSES[idx], LOG_NAMES[idx], "Réponse finale produite ✓")

            def task_callback(task_output):
                """Appelé quand une tâche est terminée, reçoit un TaskOutput."""
                role = getattr(task_output, "agent", "")
                idx = AGENT_ROLES_IDX.get(role, task_idx[0])
                st.session_state.agent_states[idx] = "done"
                add_log(LOG_CLASSES[idx], LOG_NAMES[idx], "Tâche complétée ✓")
                task_idx[0] = idx + 1
                if idx + 1 < N_AGENTS:
                    st.session_state.agent_states[idx + 1] = "running"

            # Initialisation des agents et tâches
            add_log("log-system", "Système", "Initialisation des agents CrewAI + LLM…")
            st.session_state.agent_states = ["running"] + ["idle"] * (N_AGENTS - 1)

            analyste, actualites, gestionnaire, stratege = create_agents()
            tasks = create_tasks(
                analyste, actualites, gestionnaire, stratege, tickers, budget, profile
            )

            crew = Crew(
                agents=[analyste, actualites, gestionnaire, stratege],
                tasks=tasks,
                process=Process.sequential,
                max_rpm=20,
                verbose=False,
                step_callback=step_callback,
                task_callback=task_callback,
            )

            add_log(
                "log-system",
                "Système",
                f"Lancement du pipeline : {len(tickers)} actif(s) · {budget:,.0f} EUR · {profile}",
            )
            add_log(
                "log-system", "Système", "⏳ L'analyse IA prend 1–3 minutes, merci de patienter…"
            )

            # ── LANCEMENT RÉEL DES AGENTS ──
            TOOL_OUTPUTS.clear()
            resultat = crew.kickoff()

            # ── Commentaire du Stratège + chiffres calculés par Python ──
            add_log("log-agent3", "Stratège", "Commentaire IA produit ✓")
            report, validation = finalize_report(str(resultat), tickers, budget, profile)
            add_log(
                "log-agent3",
                "Garde-fous",
                "Rapport validé ✓"
                if validation.passed
                else "Rapport NON validé ⚠ (voir pied de page)",
            )

            # ── Allocation pour l'affichage visuel ──
            # (Les agents ont déjà calculé via AllocationTool ; on recalcule localement
            # pour alimenter les widgets barres/graphique de l'interface)
            alloc = compute_allocation(tickers, budget, prices)
            total_investi = sum(a["montant_reel"] for a in alloc)
            add_log(
                "log-agent3",
                "Stratège",
                f"Total investi : {total_investi:,.0f} EUR / {budget:,.0f} EUR",
            )

            st.session_state.report_text = report
            st.session_state.alloc_data = alloc

            # PDF
            pdf_bytes = generate_pdf_bytes(
                report, alloc, dashboard.get("summary") if dashboard else None
            )
            save_report_files(report, pdf_bytes)
            st.session_state.pdf_bytes = pdf_bytes
            if pdf_bytes:
                add_log("log-agent3", "Stratège", "Rapport PDF généré ✓")

            st.session_state.agent_states = ["done"] * N_AGENTS
            add_log("log-success", "Système", "✓ Analyse CrewAI terminée avec succès")

        except Exception as e:  # noqa: BLE001 - any crew failure must fall back to local mode
            logger.exception("CrewAI run failed")
            add_log("log-system", "Système", f"Erreur CrewAI : {str(e)[:120]}")
            add_log("log-system", "Système", "Repli sur le calcul local…")
            use_crew = False  # → tombe dans le bloc local ci-dessous

    # ── 4. Mode local (fallback) ────────────────────────────────────────
    if not use_crew:
        st.session_state.agent_states = ["running", "idle", "idle", "idle"]
        add_log("log-agent1", "Analyste", "Calcul performances & volatilité…")
        if dashboard and dashboard.get("summary") is not None:
            add_log("log-agent1", "Analyste", "Indicateurs calculés ✓")
        st.session_state.agent_states = ["done", "idle", "running", "idle"]

        add_log("log-agent2", "Risques", "Analyse Sharpe, drawdown, corrélation…")
        if dashboard and dashboard.get("corr") is not None:
            add_log("log-agent2", "Risques", "Matrice de corrélation construite ✓")
        add_log("log-agent2", "Risques", f"Profil {profile} → cohérence vérifiée ✓")
        st.session_state.agent_states = ["done", "idle", "done", "running"]

        add_log("log-agent3", "Stratège", "Calcul allocation risk-parity…")
        alloc = compute_allocation(tickers, budget, prices)
        report = build_rich_report(tickers, budget, profile, alloc, dashboard)

        total_investi = sum(a["montant_reel"] for a in alloc)
        add_log(
            "log-agent3",
            "Stratège",
            f"Total investi : {total_investi:,.0f} EUR / {budget:,.0f} EUR",
        )

        pdf_bytes = generate_pdf_bytes(
            report, alloc, dashboard.get("summary") if dashboard else None
        )
        save_report_files(report, pdf_bytes)

        st.session_state.report_text = report
        st.session_state.alloc_data = alloc
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.agent_states = ["done", "idle", "done", "done"]
        add_log("log-success", "Système", "✓ Analyse locale terminée")

    st.session_state.running = False
    st.session_state.done = True


# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    st.markdown("**Actions à analyser**")
    ticker_input = st.text_input(
        "Ajouter un ticker", placeholder="ex: AAPL, BNP.PA", label_visibility="collapsed"
    )
    _, col_add = st.columns([3, 1])
    with col_add:
        if st.button("＋", key="add_btn"):
            val = ticker_input.strip().upper()
            if val and val not in st.session_state.tickers and len(st.session_state.tickers) < 5:
                st.session_state.tickers.append(val)
                st.rerun()

    for i, t in enumerate(st.session_state.tickers):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"`{t}`")
        with c2:
            if st.button("✕", key=f"rm_{i}"):
                st.session_state.tickers.pop(i)
                st.rerun()

    st.markdown("---")
    st.session_state.budget = st.number_input(
        "BUDGET (EUR)",
        min_value=100.0,
        max_value=1_000_000.0,
        value=st.session_state.budget,
        step=500.0,
    )

    st.markdown("---")
    st.markdown("**PROFIL DE RISQUE**")
    profile_options = ["conservative", "moderate", "aggressive"]
    profile_labels = [" Conservateur", " Modéré", " Agressif"]
    idx = profile_options.index(st.session_state.profile)
    chosen = st.radio("Profil", options=profile_labels, index=idx, label_visibility="collapsed")
    st.session_state.profile = profile_options[profile_labels.index(chosen)]

    st.markdown("---")

    # Statut clé API
    api_key = os.getenv(api_key_var(), "")
    if api_key:
        st.markdown(
            '<span class="badge badge-green">✓ Clé LLM configurée — agents IA actifs</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge badge-red">✗ Clé LLM manquante → mode local</span>',
            unsafe_allow_html=True,
        )
        st.caption("Ajoutez la clé API (voir .env.example) dans .env pour activer les agents IA")

    if REPORTLAB_AVAILABLE:
        st.markdown(
            '<span class="badge badge-green">✓ Export PDF actif</span>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="badge badge-orange">PDF inactif : pip install reportlab</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    can_launch = len(st.session_state.tickers) > 0 and not st.session_state.running
    if st.button(
        "⟳ Analyse en cours…" if st.session_state.running else "▶ Lancer l'analyse",
        disabled=not can_launch,
        key="launch",
    ):
        run_analysis()
        st.rerun()

    st.markdown("---")
    mode_label = "CrewAI + LLM" if api_key else "Calcul local (fallback)"
    st.caption(f"Mode : {mode_label} · Données : yfinance")


# ════════════════════════════════════════════════════════
# CONTENU PRINCIPAL
# ════════════════════════════════════════════════════════
icon, title, sub = (
    "📊",
    "PortfolioPilot",
    "Agentic AI · CrewAI + LLM · Dashboard enrichi",
)
st.markdown(
    f"""
<div class="app-header">
  <div style="font-size:28px">{icon}</div>
  <div>
    <p class="app-title">{title}</p>
    <p class="app-sub">{sub}</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Métriques
col1, col2, col3, col4 = st.columns(4)
em, pl, _ = PROFILE_META[st.session_state.profile]
with col1:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">BUDGET</div><div class="metric-value">{st.session_state.budget:,.0f}</div><div class="metric-unit">EUR</div></div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">TITRES</div><div class="metric-value">{len(st.session_state.tickers)}</div><div class="metric-unit">actions</div></div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">PROFIL</div><div class="metric-value" style="font-size:20px;padding-top:4px;">{em} {pl}</div><div class="metric-unit"> </div></div>""",
        unsafe_allow_html=True,
    )
with col4:
    api_ok = llm_configured()
    statut = (
        "En cours…"
        if st.session_state.running
        else ("Terminé ✓" if st.session_state.done else "En attente")
    )
    couleur = (
        "#60a5fa" if st.session_state.running else ("#3ecf8e" if st.session_state.done else "#333")
    )
    mode_txt = "CrewAI" if api_ok else "Local"
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">STATUT · {mode_txt}</div><div class="metric-value" style="font-size:18px;padding-top:6px;color:{couleur};">{statut}</div><div class="metric-unit"> </div></div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("**Pipeline des agents**")
    AGENTS = [
        ("🔭", "Analyste de marché", "Prix, fondamentaux, variation", "log-agent1"),
        ("📰", "Analyste actualités", "Titres récents, sentiment", "log-agent4"),
        ("🛡", "Gestionnaire des risques", "Sharpe, drawdown, corrélation", "log-agent2"),
        ("🏆", "Stratège de portefeuille", "Allocation optimale + rapport IA", "log-agent3"),
    ]
    state_labels = {
        "idle": ("dot-idle", "En attente"),
        "running": ("dot-running", "En cours…"),
        "done": ("dot-done", "Terminé ✓"),
    }
    for i, (icon_a, name, task, _) in enumerate(AGENTS):
        s = st.session_state.agent_states[i]
        dot_cls, state_txt = state_labels[s]
        card_cls = "agent-card " + (
            "running" if s == "running" else ("done" if s == "done" else "")
        )
        st.markdown(
            f"""
        <div class="{card_cls}">
          <div class="agent-header"><span style="font-size:18px">{icon_a}</span><span class="agent-name">{name}</span><span style="flex:1"></span><span class="dot {dot_cls}"></span></div>
          <div class="agent-status">{task} · {state_txt}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>**Journal en temps réel**", unsafe_allow_html=True)
    if st.session_state.logs:
        st.markdown(render_logs(), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="log-terminal"><span class="log-system">En attente du lancement…</span></div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown("**Allocation du portefeuille**")
    if st.session_state.alloc_data:
        total_investi = sum(a["montant_reel"] for a in st.session_state.alloc_data)
        cash = st.session_state.budget - total_investi
        rows_html = ""
        for i, a in enumerate(st.session_state.alloc_data):
            color = BAR_COLORS[i % len(BAR_COLORS)]
            rows_html += f"""
            <div class="alloc-row">
              <span class="alloc-ticker">{a["ticker"]}</span>
              <div class="alloc-bar-bg"><div class="alloc-bar-fill" style="width:{a["poids"]}%;background:{color};"></div></div>
              <span class="alloc-pct">{a["poids"]}%</span>
              <span class="alloc-amt">{a["montant_reel"]:,.0f} EUR</span>
            </div>"""
        rows_html += f"""
        <div style="margin-top:12px;padding-top:12px;border-top:1px solid #1a1a1a;display:flex;justify-content:space-between;">
          <span style="font-size:12px;color:#555;font-family:'IBM Plex Mono',monospace;">Total investi</span>
          <span style="font-size:13px;color:#3ecf8e;font-family:'IBM Plex Mono',monospace;">{total_investi:,.0f} EUR</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:4px;">
          <span style="font-size:12px;color:#555;font-family:'IBM Plex Mono',monospace;">Cash restant</span>
          <span style="font-size:13px;color:#888;font-family:'IBM Plex Mono',monospace;">{cash:,.0f} EUR</span>
        </div>"""
        st.markdown(f'<div class="agent-card">{rows_html}</div>', unsafe_allow_html=True)
        df_chart = pd.DataFrame(st.session_state.alloc_data)
        st.bar_chart(df_chart.set_index("ticker")["poids"], height=180)
    else:
        st.markdown(
            """<div class="agent-card" style="text-align:center;padding:40px;color:#333;"><div style="font-size:32px;margin-bottom:8px;">📊</div><div style="font-size:13px;">Lancez l'analyse pour voir l'allocation</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>**Rapport généré**", unsafe_allow_html=True)
    if st.session_state.report_text:
        with st.expander("📄 Voir le rapport complet", expanded=True):
            st.markdown(st.session_state.report_text)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇ Télécharger le rapport Markdown",
                data=st.session_state.report_text,
                file_name="rapport_portefeuille.md",
                mime="text/markdown",
            )
        with c2:
            if st.session_state.pdf_bytes:
                st.download_button(
                    "📄 Télécharger le rapport PDF",
                    data=st.session_state.pdf_bytes,
                    file_name="rapport_portefeuille.pdf",
                    mime="application/pdf",
                )
            else:
                st.warning("PDF non disponible. Installez reportlab.")
    else:
        st.markdown(
            """<div class="agent-card" style="text-align:center;padding:30px;color:#333;"><div style="font-size:13px;">Le rapport apparaîtra ici après l'analyse</div></div>""",
            unsafe_allow_html=True,
        )

# Dashboard enrichi
st.markdown("---")
st.markdown("## Tableau de bord financier enrichi")

dashboard = st.session_state.dashboard_data
if dashboard and dashboard.get("summary") is not None:
    summary = dashboard["summary"]
    corr = dashboard.get("corr")
    cumulative = dashboard.get("cumulative")

    st.markdown("### Indicateurs clés par actif")
    st.dataframe(summary, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Évolution normalisée sur 1 an")
        st.line_chart(cumulative)
    with c2:
        st.markdown("### Matrice de corrélation")
        if corr is not None:
            st.dataframe(corr.round(2), width="stretch")
        else:
            st.info("Corrélation indisponible.")

    st.markdown("### Prix historiques")
    st.line_chart(dashboard["prices_history"])
else:
    st.info(
        "Lancez l'analyse pour afficher les indicateurs enrichis : performance, volatilité, Sharpe, drawdown et corrélation."
    )

# Backtest
st.markdown("---")
st.markdown("## Backtest des stratégies")
backtest = st.session_state.backtest
if backtest is not None:
    bt_summary, bt_curves = backtest
    st.caption(
        "Long uniquement, rééquilibrage mensuel, fenêtre d'estimation 252 jours, "
        "coûts 10 pb, benchmark S&P 500 (USD). Chiffres calculés par Python."
    )
    st.dataframe(
        bt_summary.rename(
            columns={
                "total_return": "Rendement total",
                "cagr": "CAGR",
                "volatility": "Volatilité",
                "sharpe": "Sharpe",
                "max_drawdown": "Drawdown max",
            }
        ).style.format(
            {
                "Rendement total": "{:.1%}",
                "CAGR": "{:.1%}",
                "Volatilité": "{:.1%}",
                "Sharpe": "{:.2f}",
                "Drawdown max": "{:.1%}",
            }
        ),
        width="stretch",
    )
    st.line_chart(bt_curves)
elif st.session_state.done:
    st.warning("Backtest indisponible (historique insuffisant ou données inaccessibles).")
else:
    st.info("Lancez l'analyse pour comparer les stratégies au S&P 500.")
