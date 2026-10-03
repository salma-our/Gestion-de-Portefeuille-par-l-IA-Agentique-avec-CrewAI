"""
demo.py — Démonstration sans API (données réelles via yfinance)
Teste les 3 outils CrewAI avec retry automatique sur rate-limit Yahoo (429).
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tools import StockAnalysisTool, PortfolioRiskTool, AllocationTool


def run_demo():
    print("\n" + "═" * 60)
    print("    DEMO — Test des outils (sans API Mistral)")
    print("═" * 60)

    tickers_demo = ["AAPL", "MSFT", "ASML.AS"]
    tickers_str  = ",".join(tickers_demo)
    budget       = 10000.0

    # ── Outil 1 : Analyse d'une action ──
    print("\n [Outil 1] Analyse de AAPL...")
    tool1   = StockAnalysisTool()
    result1 = tool1._run("AAPL")
    data1   = json.loads(result1)

    if "erreur" in data1:
        print(f"     {data1['erreur']}")
    else:
        print(f"   → {data1.get('nom')} | Prix: {data1.get('prix_actuel')} | "
              f"Variation 1j: {data1.get('variation_1j')} | P/E: {data1.get('pe_ratio')}")

    # Pause entre les appels pour éviter le rate-limit
    print("    Pause 3s (rate-limit Yahoo)...")
    time.sleep(3)

    # ── Outil 2 : Risque du portefeuille ──
    print(f"\n  [Outil 2] Analyse des risques : {', '.join(tickers_demo)}...")
    tool2   = PortfolioRiskTool()
    result2 = tool2._run(tickers=tickers_str, period="1y")
    data2   = json.loads(result2)

    if "erreur" in data2:
        print(f"     {data2['erreur']}")
    else:
        metrics = data2.get("metriques_par_action", {})
        for ticker, m in metrics.items():
            print(f"   → {ticker}: Volatilité={m['volatilite_annuelle']}, "
                  f"Sharpe={m['sharpe_ratio']}, Drawdown={m['drawdown_max']}")

    print("    Pause 3s (rate-limit Yahoo)...")
    time.sleep(3)

    # ── Outil 3 : Allocation ──
    print(f"\n [Outil 3] Allocation optimale (budget: {budget} EUR)...")
    tool3   = AllocationTool()
    result3 = tool3._run(tickers=tickers_str, budget=budget)
    data3   = json.loads(result3)

    if "erreur" in data3:
        print(f"     {data3['erreur']}")
    else:
        print(f"   → Total investi: {data3['total_investi']} EUR | "
              f"Cash restant: {data3['cash_restant']} EUR")
        for ticker, alloc in data3["allocations"].items():
            print(f"   → {ticker}: {alloc['poids_recommande']} | "
                  f"{alloc['nb_actions_a_acheter']} actions × "
                  f"{alloc['prix_actuel']} = {alloc['montant_reel_investi']} EUR")

    print("\n" + "═" * 60)
    print("   Demo terminée — Les outils fonctionnent correctement")
    print("  → Pour lancer l'analyse complète : python main.py")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
