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
        En vous appuyant sur l'analyse de marché et l'évaluation des risques, rédigez la partie
        QUALITATIVE du rapport destiné à l'investisseur.

        Utilisez les outils portfolio_allocation (tickers: "{tickers_str}", budget: {budget}) et
        portfolio_backtest (tickers: "{tickers_str}") pour comprendre l'allocation et la
        comparaison des stratégies.

        Le système génère automatiquement, à l'identique des outils, tous les tableaux chiffrés
        (données par action, risque, allocation, cash restant, backtest). Ne les reproduisez pas.

        Rédigez en français, en markdown, EXACTEMENT ces cinq sections :
        ### Résumé exécutif
        ### Lecture des actions
        (forces et faiblesses de chaque titre, exprimées en mots)
        ### Stratégie d'investissement
        (horizon, fréquence de rééquilibrage, conditions de renforcement et de réduction,
        décrites qualitativement)
        ### Risques à surveiller
        (marché, secteur, risques propres à chaque titre)
        ### Conclusion
        (recommandation finale claire et prochaine étape)

        RÈGLES (contrôlées automatiquement, tout manquement invalide le rapport) :
        - N'écrivez AUCUN nombre financier : ni pourcentage, ni montant, ni décimale, ni ratio,
          ni taux de change. Exprimez les idées en mots (élevé, faible, supérieur au benchmark).
        - Ne calculez rien et n'inventez aucun seuil chiffré.
        - Si une information manque, écrivez "donnée indisponible".

        Profil de risque de l'investisseur : {profil_risque}
        """,
        expected_output="""
        Un commentaire markdown en français contenant exactement les cinq sections demandées
        (Résumé exécutif, Lecture des actions, Stratégie d'investissement, Risques à surveiller,
        Conclusion), sans aucun chiffre.
        """,
        agent=stratege,
        context=[tache_analyse, tache_risque],
    )

    return [tache_analyse, tache_risque, tache_strategie]
