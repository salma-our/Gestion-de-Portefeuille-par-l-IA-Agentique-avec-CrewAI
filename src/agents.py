"""agents.py — The 3 CrewAI agents (analyst, risk manager, strategist)."""

from crewai import Agent

from src.llm import get_llm
from src.tools import analyze_portfolio_risk, analyze_stock, calculate_optimal_allocation


def create_agents():
    """Crée et retourne les 3 agents du système."""

    llm = get_llm()

    # ── Agent 1 : Analyste de Marché ────────────────────────
    analyste_marche = Agent(
        role="Analyste de Marché",
        goal=(
            "Analyser les actions boursières sélectionnées et produire "
            "un rapport détaillé sur leurs fondamentaux, performances et tendances."
        ),
        backstory=(
            "Vous êtes un analyste financier senior avec 15 ans d'expérience sur les marchés "
            "européens et américains. Expert en analyse fondamentale et technique. "
            "⚠️ RÈGLE STRICTE : Toutes les données chiffrées viennent de yfinance. "
            "Si une donnée est indisponible, vous indiquez 'donnée indisponible' au lieu d'inventer. "
            "N'INVENTEZ JAMAIS de chiffres financiers, même pour combler des lacunes."
        ),
        tools=[analyze_stock],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        max_retry_limit=1,
    )

    # ── Agent 2 : Gestionnaire des Risques ──────────────────
    gestionnaire_risques = Agent(
        role="Gestionnaire des Risques",
        goal=(
            "Évaluer les risques du portefeuille : volatilité, corrélation, "
            "drawdown et compatibilité avec le profil de risque."
        ),
        backstory=(
            "Vous êtes un spécialiste en gestion des risques quantitatifs, "
            "ancien gérant de fonds institutionnels. Votre mission est de protéger le capital "
            "tout en optimisant le ratio risque/rendement. "
            "⚠️ RÈGLE STRICTE : Chaque métrique (Sharpe, volatilité, drawdown) "
            "provient d'un calcul Python sur des données réelles. "
            "Ne JAMAIS estimer ou inventer une métrique de risque."
        ),
        tools=[analyze_portfolio_risk],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        max_retry_limit=1,
    )

    # ── Agent 3 : Stratège de Portefeuille ──────────────────
    stratege_portefeuille = Agent(
        role="Stratège de Portefeuille",
        goal=(
            "Synthétiser les analyses et recommander une allocation optimale, "
            "avec des montants précis à investir selon le budget et le profil."
        ),
        backstory=(
            "Vous êtes un stratège d'investissement reconnu, ancien gérant chez BNP Paribas. "
            "Vous combinez analyses fondamentales et quantitatives pour construire des portefeuilles "
            "performants et adaptés à chaque investisseur. "
            "Vous rédigez des rapports clairs, actionnables et pédagogiques. "
            "⚠️ RÈGLE STRICTE : Les allocations viennent de calculs exacts (risk-parity, Markowitz). "
            "Les montants sont basés sur les prix réels de yfinance. "
            "N'INVENTEZ JAMAIS d'allocation 'de bon sens' sans calcul justifiable."
        ),
        tools=[calculate_optimal_allocation],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        max_retry_limit=1,
    )

    return analyste_marche, gestionnaire_risques, stratege_portefeuille
