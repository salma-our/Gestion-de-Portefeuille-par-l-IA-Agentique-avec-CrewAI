# Gestionnaire de Portefeuille Agentic AI

> **CrewAI + Mistral AI (gratuit) + yfinance**  
> 3 agents IA spécialisés pour analyser et optimiser votre portefeuille boursier

---

## ⚠️ Important : Intégrité des données financières

**Règle stricte** : Les agents IA n'inventent **JAMAIS** de chiffres financiers.
- Toutes les données chiffrées proviennent de **yfinance** ou de calculs Python testés
- Si une donnée est indisponible → l'agent l'indique explicitement ("donnée indisponible")
- Pas de fallback sur "connaissances générales" pour les métriques

*Ce projet respecte les standards AMF/ESMA : transparence et traçabilité des données.*

---

## Architecture du projet

```
portfolio_crew_fixed/
│
├── main.py                  # Point d'entrée principal (CLI)
├── app.py                   # Interface web Streamlit
├── demo.py                  # Test sans clé API (données réelles yfinance)
├── requirements.txt         # Dépendances Python
├── .env.example             # Template de configuration
├── .env                     # ← À créer (votre clé API)
│
├── src/
│   ├── __init__.py
│   ├── agents.py            # Définition des 3 agents CrewAI + Mistral
│   ├── tasks.py             # Tâches séquentielles des agents
│   └── tools.py             # 3 outils : analyse, risque, allocation
│
├── reports/
│   ├── report_pdf.py        # Conversion markdown → PDF
│   └── rapport_portefeuille.md   # ← Généré après analyse (git-ignored)
│
└── .gitignore               # Exclusions git (.env, reports/, __pycache__)
```

---

## Les 3 Agents


|
| **Analyste de Marché** | Analyse fondamentale de chaque action (prix, P/E, bêta, dividende) | `StockAnalysisTool` |
| **Gestionnaire des Risques** | Calcule volatilité, Sharpe ratio, corrélations, drawdown max | `PortfolioRiskTool` |
| **Stratège de Portefeuille** | Synthèse finale + allocation optimale (risk-parity) | `AllocationTool` |

### Pipeline séquentiel

```
[Analyste Marché] → [Gestionnaire Risques] → [Stratège Portefeuille]
     Analyse              Évaluation              Rapport final
  fondamentale            des risques           + Allocation
```

---

##  Installation

### 1. Cloner / télécharger le projet

```bash
cd portfolio_crew_fixed
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API Mistral (gratuite)

1. Aller sur [console.mistral.ai](https://console.mistral.ai)
2. Créer un compte → **API Keys** → **Create new key**
3. Copier `.env.example` vers `.env` :

```bash
cp .env.example .env
```

4. Éditer `.env` et coller votre clé :

```env
MISTRAL_API_KEY=votre_cle_ici
MISTRAL_MODEL=mistral-small-latest
PORTFOLIO_BUDGET=10000
RISK_PROFILE=moderate
```

---

##  Utilisation

### Test rapide (sans clé API)

```bash
python demo.py
```

### Analyse complète avec les paramètres par défaut

```bash
python main.py
```

### Avec vos propres paramètres

```bash
# Portefeuille tech américain, budget 5000€, profil agressif
python main.py --tickers AAPL,MSFT,NVDA --budget 5000 --profil aggressive

# Portefeuille européen, profil conservateur
python main.py --tickers BNP.PA,TTE.PA,OR.PA --budget 20000 --profil conservative

# Mix franco-américain
python main.py --tickers AAPL,ASML.AS,AIR.PA --budget 15000 --profil moderate
```

### Portefeuilles par défaut (si --tickers non spécifié)


| `conservative` | BNP.PA, TTE.PA, OR.PA |
| `moderate` | AAPL, MSFT, ASML.AS |
| `aggressive` | NVDA, META, TSLA |

---

## 📊Ce que génère le système

Le rapport final (`reports/rapport_portefeuille.md`) contient :

1. **Résumé exécutif** — Vue d'ensemble de l'opportunité
2. **Analyse des actions** — Fondamentaux de chaque titre
3. **Profil de risque** — Adéquation avec votre objectif
4. **Tableau d'allocation** — Poids, montants, nombre d'actions
5. **Stratégie** — Horizon, points d'entrée/sortie
6. **Risques à surveiller** — Top 3 des alertes
7. **Conclusion** — Recommandation finale

---


>
##  Personnalisation

### Ajouter une action à analyser

```bash
python main.py --tickers AAPL,MSFT,GOOGL,AMZN,META --budget 25000 --profil aggressive
```

> Restez sous 5 tickers pour ne pas dépasser les limites de tokens de l'API gratuite.

### Modifier les paramètres dans `.env`

```env
PORTFOLIO_BUDGET=50000        # Votre budget
RISK_PROFILE=conservative     # conservative / moderate / aggressive
MISTRAL_MODEL=mistral-small-latest
```

---

##  Dépannage

| Erreur | Solution |
|--------|----------|
| `MISTRAL_API_KEY non définie` | Vérifier que `.env` existe et contient la clé |
| `No data found for ticker` | Vérifier le symbole sur Yahoo Finance |
| `Rate limit exceeded` | Attendre 60s, le quota Mistral gratuit est limité |
| `ModuleNotFoundError` | Relancer `pip install -r requirements.txt` |
| `Cookie/crumb fetch failed (SSLError)` | Le chemin du projet/venv contient des accents : copier `certifi/cacert.pem` dans un dossier ASCII et définir `CURL_CA_BUNDLE` dans `.env` |
| `Rate limit reached ... TPM` (Groq) | Palier gratuit limité en tokens/minute : analyser 2-3 tickers max ; les retries automatiques absorbent les pics |

### Trouver les bons symboles boursiers

- Actions US : `AAPL`, `MSFT`, `GOOGL`, `NVDA`, `META`
- Actions françaises : `BNP.PA`, `AIR.PA`, `TTE.PA`, `OR.PA`, `SAN.PA`
- Actions européennes : `ASML.AS`, `SAP.DE`, `NESN.SW`, `MC.PA`

---

## Dépendances principales

| Package | Version | Rôle |
|---------|---------|------|
| `crewai` | 0.80.0 | Framework multi-agents |
| `langchain-mistralai` | latest | Intégration LLM Mistral |
| `mistralai` | 1.2.0 | SDK officiel Mistral |
| `yfinance` | 0.2.40 | Données boursières gratuites |
| `pandas` | 2.2.2 | Calculs financiers |
| `numpy` | 1.26.4 | Calculs numériques |

---

*Projet éducatif — Ne constitue pas un conseil en investissement financier.*
