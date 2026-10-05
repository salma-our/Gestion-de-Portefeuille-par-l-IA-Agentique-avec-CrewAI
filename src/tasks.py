"""
tasks.py — Définition des tâches pour chaque agent
"""

from typing import TYPE_CHECKING

from crewai import Task

if TYPE_CHECKING:
    from crewai import Agent


def create_tasks(
    analyste: "Agent",
    gestionnaire: "Agent",
    stratege: "Agent",
    tickers: list[str],
    budget: float,
    profil_risque: str,
) -> list[Task]:
    """Crée les 3 tâches séquentielles pour le pipeline."""

    tickers_str = ",".join(tickers)
    tickers_display = ", ".join(tickers)

    # ─────────────────────────────────────────────
    # Tâche 1 : Analyse fondamentale des actions
    # ─────────────────────────────────────────────
    tache_analyse = Task(
        description=f"""
        Analysez chacune des actions suivantes : {tickers_display}

        Pour CHAQUE action, utilisez l'outil stock_analysis pour récupérer :
        - Prix actuel et variations récentes
        - Fondamentaux (P/E ratio, dividende, capitalisation)
        - Position par rapport aux plus hauts/bas 52 semaines
        - Beta (sensibilité au marché)

        Rédigez ensuite un rapport d'analyse synthétique pour chaque action,
        en mettant en évidence les forces, faiblesses et opportunités basées sur les DONNÉES RÉELLES.

        IMPORTANT : Classez les actions UNIQUEMENT selon les chiffres yfinance.
        Ne JAMAIS inventer de données. Si une donnée manque, indiquez "donnée indisponible".

        Budget de l'investisseur : {budget} EUR
        Profil de risque : {profil_risque}
        """,
        expected_output=f"""
        Un rapport d'analyse structuré contenant :
        1. Fiche résumée de chaque action ({tickers_display}) — toutes données de yfinance
        2. Points forts et points de vigilance basés UNIQUEMENT sur les chiffres réels
        3. Classement final des actions par score (P/E, Sharpe, etc.) avec justification chiffrée
        ⚠️ INTÉGRITÉ : Si une donnée manque → indiquer "donnée indisponible", ne JAMAIS estimer
        """,
        agent=analyste,
    )

    # ─────────────────────────────────────────────
    # Tâche 2 : Évaluation des risques
    # ─────────────────────────────────────────────
    tache_risque = Task(
        description=f"""
        Sur la base de l'analyse fournie par l'Analyste de Marché, évaluez les risques
        du portefeuille composé des actions : {tickers_display}

        Utilisez l'outil portfolio_risk avec les paramètres :
        - tickers: "{tickers_str}"
        - period: "1y"

        Analysez :
        1. La volatilité annuelle de chaque action
        2. Le Sharpe ratio (rendement ajusté au risque)
        3. Le drawdown maximum historique (perte potentielle)
        4. La matrice de corrélation (diversification effective ?)

        Évaluez la compatibilité de ce portefeuille avec le profil : {profil_risque}
        (conservative = faible volatilité, moderate = équilibré, aggressive = haute croissance)

        Identifiez les actions qui augmentent ou réduisent le risque global.
        """,
        expected_output="""
        Un rapport de risque contenant :
        1. Métriques de risque de chaque action (volatilité, Sharpe, drawdown)
        2. Analyse de la diversification (corrélations)
        3. Évaluation de la compatibilité avec le profil de risque
        4. Recommandations pour optimiser le profil risque/rendement
        5. Alertes sur les actions à risque élevé
        """,
        agent=gestionnaire,
        context=[tache_analyse],
    )

    # ─────────────────────────────────────────────
    # Tâche 3 : Stratégie et allocation finale
    # ─────────────────────────────────────────────
    tache_strategie = Task(
        description=f"""
        En synthétisant l'analyse de marché et l'évaluation des risques,
        construisez le portefeuille optimal pour l'investisseur.

        Utilisez l'outil portfolio_allocation avec :
        - tickers: "{tickers_str}"
        - budget: {budget}

           Puis rédigez un rapport final complet et professionnel qui contiendra :

        1. PAGE DE SYNTHÈSE
           - Objectif du portefeuille
           - Budget total
           - Profil de risque
           - Nombre d'actifs analysés
           - Recommandation globale : acheter / attendre / diversifier

        2. RÉSUMÉ EXÉCUTIF
           - Opportunité d'investissement
           - Qualité globale du portefeuille
           - Niveau de risque global
           - Points clés à retenir

        3. ANALYSE DÉTAILLÉE DES ACTIONS
           Pour chaque action :
           - Prix actuel
           - Performance récente
           - P/E ratio
           - Beta
           - Dividende si disponible
           - Points forts
           - Points faibles
           - Avis de l'agent

        4. ANALYSE DU RISQUE
           - Volatilité annuelle
           - Sharpe ratio
           - Drawdown maximum
           - Corrélation entre les actifs
           - Commentaire sur la diversification

        5. ALLOCATION RECOMMANDÉE
           Tableau avec :
           - Ticker
           - Poids recommandé
           - Montant à investir
           - Prix actuel
           - Nombre d'actions à acheter
           - Montant réellement investi

        6. CASH RESTANT
           - Montant non investi
           - Explication du cash restant

        7. STRATÉGIE D'INVESTISSEMENT
           - Horizon conseillé
           - Fréquence de rééquilibrage
           - Conditions de renforcement
           - Conditions de réduction

        8. RISQUES À SURVEILLER
           - Risque de marché
           - Risque sectoriel
           - Risque spécifique à chaque action

        9. CONCLUSION
           - Recommandation finale claire
           - Prochaine étape pour l'investisseur
        Budget total : {budget} EUR
        Profil de risque : {profil_risque}
        """,
        expected_output="""
        Un rapport de portefeuille très complet, structuré et professionnel en français.
        Le rapport doit contenir :
        - Une synthèse exécutive
        - Une analyse par action
        - Une analyse des risques
        - Un tableau d'allocation clair
        - Une stratégie d'investissement
        - Une conclusion finale
        Le rapport doit être directement exportable en PDF.
        Important : n'inventez jamais de données. Si une donnée manque, indiquez "donnée indisponible".
        """,
        agent=stratege,
        context=[tache_analyse, tache_risque],
        output_file="reports/rapport_portefeuille.md",
    )

    return [tache_analyse, tache_risque, tache_strategie]
