# CLAUDE.md — Règles de travail pour le projet Portfolio Manager

## 🎯 Objectif
Système multi-agents (CrewAI + Mistral + yfinance) de gestion de portefeuille boursier.
**But** : Portfolio solide pour un profil junior AI Engineer en finance.

---

## 📋 Règles strictes

### 1️⃣ Zéro hallucination de chiffres financiers
- **Toute donnée chiffrée provient d'un calcul Python testé ou de yfinance**
- Si une donnée est indisponible → indiquer "donnée indisponible" au lieu d'estimer
- Pas d'approximation type "on peut supposer que..."
- **Audit** : Chaque nombre dans le rapport doit être traçable

### 2️⃣ Code Python 3.11+
- Type hints obligatoires (PEP 484)
- Docstrings courtes (max 1 ligne sauf si WHY non-obvio)
- Formaté avec ruff + black
- Aucun code mort

### 3️⃣ Commits atomiques
- Convention : `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Une changement logique = un commit
- Message : 1 ligne de contexte, pas de "fixes #123" (c'est dans la PR)

### 4️⃣ Tests avant merge
- Chaque Phase → run tests locaux
- `pytest` doit passer 100%
- Type check : `mypy src/`
- Linting : `ruff check .`

### 5️⃣ Configuration & secrets
- `.env` → git-ignored ✓
- `.env.example` → template sans clés ✓
- Mistral API : clé jamais loggée
- Docker : secrets via DOCKER_BUILDKIT

---

## 🔧 Stack technique

| Composant | Version | Notes |
|---|---|---|
| CrewAI | 0.80.0 → 0.90+ (Phase 2) | Multi-agents, LLM agnostic |
| Mistral | Free tier (1M tokens/mois) | LLM en Phase 2+ |
| yfinance | 0.2.31+ | Données boursières (Yahoo Finance) |
| Python | 3.11+ | Modern, typed |
| Streamlit | 1.35+ | UI web |

---

## 🏗️ Structure attendue

```
portfolio_crew_fixed/
├── .env.example           # Template (sans clé)
├── .gitignore             # ✓ Exclusions
├── requirements.txt       # Dependencies
├── CLAUDE.md              # Ce fichier
├── README.md              # Docs utilisateur
├── README.fr.md           # (optionnel) Français
│
├── main.py                # CLI
├── app.py                 # Streamlit UI
├── demo.py                # Test des outils
│
├── src/
│   ├── agents.py          # 3 agents + docstrings strictes
│   ├── tasks.py           # Pipeline séquentiel
│   ├── tools.py           # 3 outils yfinance
│   └── __init__.py
│
├── reports/
│   ├── report_pdf.py      # Markdown → PDF
│   └── .gitkeep           # (md/pdf git-ignored)
│
└── tests/                 # (Phase 2)
    ├── test_tools.py      # Unit tests
    ├── test_agents.py     # Agent mocking
    └── test_allocation.py # Allocation logic
```

---

## 📊 Phases du projet

### Phase 1 ✅ Hygiène (2–3j)
- `.gitignore`, `.env.example`
- README correction
- Docstrings stricte
- Remove `__pycache__`

### Phase 2 🔧 Refactor + Tests (3–5j)
- Upgrade crewai 0.90+, remove patch
- Type hints complets
- Suite de tests
- CI/GitHub Actions

### Phase 3 📈 Backtesting (4–6j)
- Markowitz mean-variance
- Historical simulation
- Benchmark vs S&P500/CAC40

### Phase 4 🤖 Agents avancés (5–7j)
- Sentiment analysis agent
- Risk controller guardrails
- RAG (optional)

### Phase 5 🚀 Déploiement (3–4j)
- Docker + docker-compose
- Streamlit Cloud / Railway
- FastAPI (optional)

---

## 🧪 Testing checklist

Avant chaque commit :

```bash
# Linting
ruff check .
black --check .

# Type hints
mypy src/

# Tests
pytest tests/ -v

# Manual
python demo.py  # Outils sans API
python main.py --tickers AAPL,MSFT --budget 5000  # CLI complet
streamlit run app.py  # UI
```

---

## 🎓 Profil cible : Junior AI Engineer

Ce projet démontre :
1. **Architecture multi-agents** (CrewAI)
2. **Intégrité des données** (zéro hallucination chiffrée)
3. **Finance réelle** (yfinance, metrics: Sharpe, volatilité, drawdown)
4. **Python production** (typing, tests, CI/CD)
5. **UI moderne** (Streamlit + CLI + API optionnelle)

**Pour un recruteur** : Montre la capacité à construire un système IA **trustworthy** et **auditable**.

---

## 📞 Questions fréquentes

**Q: Mistral API ne répond pas?**  
A: Fallback auto sur calcul local (app.py:584). Vérifier quota 1M tokens/mois.

**Q: yfinance returns "429 Too Many Requests"?**  
A: Retry auto × 3 avec sleep. Si ça persiste → utiliser `pandas-datareader` ou autre source.

**Q: Comment déboguer un agent?**  
A: Activer `verbose=True` dans agents.py, lancer `python main.py --tickers AAPL --budget 1000`.

**Q: Peut-on ajouter OpenAI à la place de Mistral?**  
A: Oui, Phase 2. Changer `agents.py:17` → `"openai/gpt-4-turbo"` + adapters.

---

**Mis à jour** : 2026-10-04  
**Statut** : Phase 1 en cours
