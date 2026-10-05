# CLAUDE.md — PortfolioPilot working rules

Multi-agent portfolio analysis (CrewAI + LLM + yfinance). Goal: a solid portfolio project for a junior
AI Engineer profile in finance.

## Rules

- **The LLM never computes a number.** Every figure comes from a tested Python function; missing data is
  reported as `donnée indisponible`, never estimated.
- Before any significant change: propose a short plan and wait for validation.
- Small steps, one atomic commit per logical change (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
- Python 3.11+, typed, short docstrings, formatted with ruff. Code, comments and names in English;
  `README.md` in English, `README.fr.md` in French.
- Never commit an API key or a `.env` file.
- After each step: run `ruff check .`, `ruff format --check .` and `pytest`, and summarise in 5 lines max.
- Never push without an explicit user request.

## Layout

- `src/finance.py`, `optimization.py`, `strategies.py`, `backtest.py`: pure calculations, fully tested offline.
- `src/tools.py`: yfinance I/O and JSON formatting for the agents (4 tools, EUR conversion via Yahoo FX).
- `src/facts.py`, `report_builder.py`, `pipeline.py`: numeric report sections rendered in Python, LLM narrative
  (no figures) appended, then validated by `src/guardrails.py`.
- `src/agents.py`, `tasks.py`, `llm.py`: agents, sequential tasks, provider config (`LLM_MODEL`).
- `main.py` (CLI), `app.py` (Streamlit), `demo.py` (tools only).
- `evaluation/`: offline scenarios (`python -m evaluation.run`), also replayed by `tests/test_eval.py`.
- `tests/`: pytest suite with synthetic data, no network.

## Environment notes

- Keep the venv in an ASCII-only path (curl cannot read the CA bundle under paths with accents), or set
  `CURL_CA_BUNDLE`.
- Groq free tier: 8,000 tokens per minute; run the pipeline with 2-3 tickers.

## Roadmap

1. Repository hygiene: done.
2. Refactoring, tests, CI: done.
3. Optimization and backtest: done (min variance, max Sharpe, risk parity, S&P 500 benchmark).
4. Advanced agents: guardrails and Python-rendered report done; offline evaluation scenarios done; news/sentiment, RAG to do.
5. Interface polish, Docker, deployment.
