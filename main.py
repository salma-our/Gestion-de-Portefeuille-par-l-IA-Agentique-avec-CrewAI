"""
main.py — Point d'entrée principal
Gestion de Portefeuille avec CrewAI + LLM

Usage:
    python main.py
    python main.py --tickers AAPL,MSFT,GOOGL --budget 5000 --profil moderate
"""

import argparse
import os
import sys
from datetime import datetime

from crewai import Crew, Process
from dotenv import load_dotenv

from reports.report_pdf import markdown_to_pdf

# Charger les variables d'environnement
load_dotenv()

# Vérification de la clé API
from src.llm import api_key_var, llm_configured, model_name

if not llm_configured():
    print(f" ERREUR : {api_key_var()} non définie dans le fichier .env")
    print("   → Copiez .env.example vers .env et ajoutez votre clé.")
    print("   → Clé Groq gratuite : https://console.groq.com")
    sys.exit(1)

from src.agents import create_agents
from src.guardrails import format_validation
from src.pipeline import finalize_report
from src.tasks import create_tasks
from src.tools import TOOL_OUTPUTS

# ─────────────────────────────────────────────
# Portefeuilles prédéfinis par profil de risque
# ─────────────────────────────────────────────
PORTFOLIOS_DEFAUT = {
    "conservative": ["BNP.PA", "TTE.PA", "OR.PA"],  # Grandes caps françaises stables
    "moderate": ["AAPL", "MSFT", "ASML.AS"],  # Tech solide + Europe
    "aggressive": ["NVDA", "META", "TSLA"],  # Croissance haute volatilité
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PortfolioPilot — multi-agent portfolio analysis (CrewAI)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Actions à analyser, séparées par des virgules (ex: AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Budget total à investir en EUR (défaut: depuis .env ou 10000)",
    )
    parser.add_argument(
        "--profil",
        type=str,
        choices=["conservative", "moderate", "aggressive"],
        default=None,
        help="Profil de risque : conservative / moderate / aggressive",
    )
    return parser.parse_args()


def afficher_banniere():
    print("\n" + "═" * 60)
    print("    PortfolioPilot — Agentic portfolio analysis")
    print("    Powered by CrewAI + LLM (gratuit)")
    print("═" * 60)
    print(f"    Démarrage : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("═" * 60 + "\n")


def main():
    os.makedirs("reports", exist_ok=True)
    afficher_banniere()

    args = parse_arguments()

    # ── Récupération des paramètres ──
    profil = args.profil or os.getenv("RISK_PROFILE", "moderate")
    budget = args.budget or float(os.getenv("PORTFOLIO_BUDGET", "10000"))

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        tickers = PORTFOLIOS_DEFAUT.get(profil, PORTFOLIOS_DEFAUT["moderate"])
        print(f"ℹ️  Aucun ticker spécifié → utilisation du portefeuille {profil} par défaut")

    print(f"   Portefeuille  : {', '.join(tickers)}")
    print(f"   Budget        : {budget:,.0f} EUR")
    print(f"   Profil risque : {profil.upper()}")
    print(f"   Modèle IA     : {model_name()}")
    print("\n" + "─" * 60)

    # ── Création des agents et tâches ──
    print("\nInitialisation des agents...")
    analyste, gestionnaire, stratege = create_agents()

    print(" Création des tâches...")
    tasks = create_tasks(analyste, gestionnaire, stratege, tickers, budget, profil)

    # ── Création et lancement du Crew ──
    crew = Crew(
        agents=[analyste, gestionnaire, stratege],
        tasks=tasks,
        process=Process.sequential,  # Séquentiel : analyse → risque → stratégie
        max_rpm=20,  # free-tier rate limit
        verbose=True,
    )

    print("\n Lancement de l'analyse par les agents...\n")
    print("═" * 60)

    TOOL_OUTPUTS.clear()
    resultat = crew.kickoff()
    md_path = "reports/rapport_portefeuille.md"
    pdf_path = "reports/rapport_portefeuille.pdf"

    print("\n Assemblage du rapport (chiffres calculés par Python)...")
    markdown_text, validation = finalize_report(str(resultat), tickers, budget, profil)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(format_validation(validation))
    markdown_to_pdf(markdown_text, pdf_path)

    print(f" Rapport PDF généré dans : {pdf_path}")

    # ── Affichage du résultat ──
    print("\n" + "═" * 60)
    print("    ANALYSE TERMINÉE")
    print("═" * 60)
    print("\n Rapport sauvegardé dans : reports/rapport_portefeuille.md\n")
    print("─" * 60)
    print("\n RÉSUMÉ DU STRATÈGE :\n")
    print(resultat)
    print("\n" + "═" * 60)


if __name__ == "__main__":
    main()
