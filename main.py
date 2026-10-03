"""
main.py — Point d'entrée principal
Gestion de Portefeuille avec CrewAI + Mistral AI

Usage:
    python main.py
    python main.py --tickers AAPL,MSFT,GOOGL --budget 5000 --profil moderate
"""
from reports.report_pdf import markdown_to_pdf
import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv
from crewai import Crew, Process

# Charger les variables d'environnement
load_dotenv()

# Vérification de la clé API
if not os.getenv("MISTRAL_API_KEY"):
    print(" ERREUR : MISTRAL_API_KEY non définie dans le fichier .env")
    print("   → Copiez .env.example vers .env et ajoutez votre clé Mistral.")
    print("   → Clé gratuite disponible sur : https://console.mistral.ai")
    sys.exit(1)

from src.agents import create_agents
from src.tasks import create_tasks


# ─────────────────────────────────────────────
# Portefeuilles prédéfinis par profil de risque
# ─────────────────────────────────────────────
PORTFOLIOS_DEFAUT = {
    "conservative": ["BNP.PA", "TTE.PA", "OR.PA"],      # Grandes caps françaises stables
    "moderate":     ["AAPL", "MSFT", "ASML.AS"],         # Tech solide + Europe
    "aggressive":   ["NVDA", "META", "TSLA"],             # Croissance haute volatilité
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=" Gestionnaire de Portefeuille Agentic AI avec CrewAI + Mistral"
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
    print("    GESTION DE PORTEFEUILLE — Agentic AI")
    print("    Powered by CrewAI + Mistral AI (gratuit)")
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
    print(f"   Modèle IA     : {os.getenv('MISTRAL_MODEL', 'mistral-small-latest')}")
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
        process=Process.sequential,   # Séquentiel : analyse → risque → stratégie
        verbose=True,
    )

    print("\n Lancement de l'analyse par les agents...\n")
    print("═" * 60)

    resultat = crew.kickoff()
    md_path = "reports/rapport_portefeuille.md"
    pdf_path = "reports/rapport_portefeuille.pdf"

    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()
        markdown_to_pdf(markdown_text, pdf_path)
    else:
        markdown_to_pdf(str(resultat), pdf_path)

    print(f" Rapport PDF généré dans : {pdf_path}")

    # ── Affichage du résultat ──
    print("\n" + "═" * 60)
    print("    ANALYSE TERMINÉE")
    print("═" * 60)
    print(f"\n Rapport sauvegardé dans : reports/rapport_portefeuille.md\n")
    print("─" * 60)
    print("\n RÉSUMÉ DU STRATÈGE :\n")
    print(resultat)
    print("\n" + "═" * 60)


if __name__ == "__main__":
    main()
