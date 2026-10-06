# PortfolioPilot

[English](README.md)

Analyse de portefeuille multi-agents avec **CrewAI**, un LLM (palier gratuit Groq par défaut) et **yfinance**.
Quatre agents spécialisés analysent des actions, lisent l'actualité, évaluent le risque et rédigent un rapport d'allocation incluant
un backtest walk-forward face au S&P 500.

> **Règle de conception : le LLM ne calcule jamais un chiffre.** Tous les chiffres d'un rapport viennent de
> fonctions Python testées (`src/finance.py`, `src/optimization.py`, `src/backtest.py`). Les sections chiffrées
> du rapport (données, risque, allocation, backtest) sont générées par Python ; le LLM rédige seulement un
> commentaire qualitatif qui ne doit contenir aucun chiffre. Si une donnée manque : `donnée indisponible`.

## Fonctionnement

```
[Analyste de marché] ──► [Analyste actualités] ──► [Gestionnaire des risques] ──► [Stratège] ──► reports/rapport_portefeuille.md/.pdf
 stock_analysis            news_sentiment            portfolio_risk                portfolio_allocation
                                                                                      portfolio_backtest
```

| Agent | Outil(s) | Résultat |
|---|---|---|
| Analyste de marché | `stock_analysis` | Prix, P/E, bêta, dividende, plus haut/bas 52 semaines |
| Analyste actualités | `news_sentiment` | Titres récents (Yahoo, RSS Google Actualités en secours) notés avec un lexique financier |
| Gestionnaire des risques | `portfolio_risk` | Volatilité, Sharpe, drawdown max, corrélations |
| Stratège de portefeuille | `portfolio_allocation`, `portfolio_backtest` | Allocation risk-parity en actions entières, backtest des stratégies |

### Garde-fous

Après chaque exécution, `src/guardrails.py` valide le rapport final avec des règles déterministes et ajoute le
résultat en pied de page : sections obligatoires présentes, chaque chiffre traçable jusqu'à une sortie d'outil,
données manquantes signalées, allocation cohérente avec le budget, et aucun chiffre dans le commentaire du LLM.
Les prix sont convertis en EUR avec les taux de change Yahoo, calculés en Python.

### Sentiment des actualités

Les titres des 30 derniers jours sont notés par Python avec un lexique financier (négations gérées) : un score
dans [-1, 1] par titre, moyenné par ticker. C'est un signal grossier, pas un modèle, et la recherche Yahoo renvoie
parfois des articles qui ne font que mentionner le ticker. Le LLM ne note ni ne recompte jamais les titres.

### Évaluation

`python -m evaluation.run` rejoue 14 scénarios hors ligne (marché synthétique déterministe, commentaire fixe,
sans LLM) : un ticker invalide, tous les tickers invalides, un seul titre, un budget inférieur au prix d'une
action, un budget nul ou énorme, un historique trop court, aucune actualité récente, des actualités très négatives, des devises mixtes (USD/EUR/GBp) et trois commentaires
fautifs (rendement inventé, taux de change inventé, conclusion absente) que les garde-fous doivent rejeter.
Chaque scénario vérifie l'absence de plantage, le verdict attendu des garde-fous, le signalement des blocs
manquants (`donnée indisponible`) et que l'allocation ne dépasse jamais le budget. Les mêmes scénarios
tournent dans `pytest`.

### Backtest

Compare les portefeuilles équipondéré, volatilité inverse, variance minimale, Sharpe maximal et risk parity
(contributions égales au risque) au S&P 500. Long uniquement, rééquilibrage mensuel, fenêtre d'estimation de
252 jours, coûts de 10 points de base. Les poids sont calculés sur le passé uniquement puis appliqués le
lendemain (pas de lookahead, vérifié par des tests). Le benchmark est coté en USD.

## Structure

```
main.py                 Point d'entrée CLI
app.py                  Interface Streamlit
demo.py                 Lance les outils sur données réelles, sans clé LLM
src/finance.py          Volatilité, rendement, Sharpe, drawdown, allocation en actions entières
src/optimization.py     Variance minimale, Sharpe max, risk parity (SLSQP, plafonds de poids)
src/strategies.py       Stratégies : fenêtre de rendements -> poids
src/backtest.py         Backtest walk-forward, coûts, comparaison au benchmark
src/tools.py            Outils CrewAI (entrées/sorties yfinance + JSON pour les agents)
src/facts.py            Exécute les outils pour collecter les chiffres du rapport
src/report_builder.py   Génère en Python les sections chiffrées du rapport
src/guardrails.py       Validation déterministe du rapport
src/pipeline.py         Assemblage : chiffres + commentaire + pied de page de validation
src/agents.py, tasks.py Agents et tâches séquentielles
src/llm.py              Configuration du fournisseur LLM
reports/report_pdf.py   Markdown -> PDF
evaluation/             Scénarios d'évaluation hors ligne et lanceur
tests/                  Suite pytest (sans réseau)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # puis ajoutez votre clé
```

Créez une clé Groq gratuite sur <https://console.groq.com> et renseignez `.env` :

```env
GROQ_API_KEY=votre_cle
LLM_MODEL=groq/openai/gpt-oss-120b
```

Tout fournisseur [LiteLLM](https://docs.litellm.ai/docs/providers) convient : changez `LLM_MODEL` (par exemple
`mistral/mistral-small-latest`) et la variable `<FOURNISSEUR>_API_KEY` correspondante.

## Utilisation

```bash
python demo.py                                                   # outils seuls, sans LLM
python main.py --tickers AAPL,MSFT --budget 5000 --profil moderate   # pipeline complet
streamlit run app.py                                             # interface web
```

Sans clé API, l'application Streamlit passe en mode local (mêmes calculs, sans texte du LLM).
Le palier gratuit Groq est limité à 8 000 tokens par minute : 2 ou 3 tickers par exécution.

## Développement

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest
```

La CI exécute les mêmes contrôles à chaque push (`.github/workflows/ci.yml`).

## Dépannage

| Symptôme | Solution |
|---|---|
| `Cookie/crumb fetch failed (SSLError)` | Le chemin du projet ou du venv contient des accents (par exemple `é`). Copiez `certifi/cacert.pem` dans un dossier sans accents et définissez `CURL_CA_BUNDLE` dans `.env`, ou déplacez le venv. |
| `Rate limit reached ... TPM` | Palier gratuit Groq : moins de tickers ; les nouvelles tentatives automatiques absorbent les pics courts. |
| `<FOURNISSEUR>_API_KEY non définie` | Vérifiez `.env` et que `LLM_MODEL` correspond à la clé renseignée. |
| Données vides pour un ticker | Vérifiez le symbole sur Yahoo Finance (`BNP.PA`, `ASML.AS`, `AAPL`). |

*Projet éducatif. Ne constitue pas un conseil en investissement.*
