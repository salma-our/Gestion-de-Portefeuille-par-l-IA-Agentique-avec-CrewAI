# PortfolioPilot

[English](README.md)

Analyse de portefeuille multi-agents avec **CrewAI**, un LLM (palier gratuit Groq par défaut) et **yfinance**.
Trois agents spécialisés analysent des actions, évaluent le risque et rédigent un rapport d'allocation incluant
un backtest walk-forward face au S&P 500.

> **Règle de conception : le LLM ne calcule jamais un chiffre.** Tous les chiffres d'un rapport viennent de
> fonctions Python testées (`src/finance.py`, `src/optimization.py`, `src/backtest.py`) exposées aux agents
> comme outils. Le LLM interprète et rédige. Si une donnée manque, le rapport indique `donnée indisponible`.

## Fonctionnement

```
[Analyste de marché] ──► [Gestionnaire des risques] ──► [Stratège] ──► reports/rapport_portefeuille.md/.pdf
 stock_analysis            portfolio_risk                portfolio_allocation
                                                         portfolio_backtest
```

| Agent | Outil(s) | Résultat |
|---|---|---|
| Analyste de marché | `stock_analysis` | Prix, P/E, bêta, dividende, plus haut/bas 52 semaines |
| Gestionnaire des risques | `portfolio_risk` | Volatilité, Sharpe, drawdown max, corrélations |
| Stratège de portefeuille | `portfolio_allocation`, `portfolio_backtest` | Allocation risk-parity en actions entières, backtest des stratégies |

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
src/agents.py, tasks.py Agents et tâches séquentielles
src/llm.py              Configuration du fournisseur LLM
reports/report_pdf.py   Markdown -> PDF
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
