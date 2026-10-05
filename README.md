# PortfolioPilot

[Français](README.fr.md)

Multi-agent portfolio analysis built with **CrewAI**, an LLM (Groq free tier by default) and **yfinance**.
Three specialised agents analyse a set of stocks, assess risk, and write an allocation report that includes a
walk-forward backtest against the S&P 500.

> **Design rule — the LLM never computes a number.** Every figure in a report comes from tested Python
> functions (`src/finance.py`, `src/optimization.py`, `src/backtest.py`). The numeric sections of the report
> (data, risk, allocation, backtest) are rendered by Python; the LLM only writes a qualitative commentary
> that must contain no figure. If data is missing, the report says `donnée indisponible`.

## How it works

```
[Market Analyst] ──► [Risk Manager] ──► [Portfolio Strategist] ──► reports/rapport_portefeuille.md/.pdf
 stock_analysis       portfolio_risk      portfolio_allocation
                                          portfolio_backtest
```

| Agent | Tool(s) | Output |
|---|---|---|
| Market Analyst | `stock_analysis` | Price, P/E, beta, dividend, 52-week range |
| Risk Manager | `portfolio_risk` | Volatility, Sharpe, max drawdown, correlations |
| Portfolio Strategist | `portfolio_allocation`, `portfolio_backtest` | Risk-parity allocation in whole shares, strategy backtest |

### Guardrails

After each run, `src/guardrails.py` validates the final report with deterministic rules and appends the
result as a footer: required sections present, every financial number traceable to a tool output,
missing data acknowledged, allocation consistent with the budget, and no figure in the LLM narrative.
Prices are converted to EUR with Yahoo FX rates computed in Python.

### Evaluation

`python -m evaluation.run` replays 12 scenarios offline (deterministic synthetic market, fixed narrative, no
LLM): one invalid ticker, all tickers invalid, a single ticker, a budget below one share, a zero or huge budget,
short price history, mixed currencies (USD/EUR/GBp), and three misbehaving narratives (invented return,
invented FX rate, missing conclusion) that the guardrails must reject. Each scenario checks that the pipeline
does not crash, that the guardrail verdict is the expected one, that missing blocks are flagged as
`donnée indisponible`, and that the allocation never exceeds the budget. The same scenarios run in `pytest`.

### Backtest

Compares equal weight, inverse volatility, minimum variance, maximum Sharpe and equal-risk-contribution
(risk parity) portfolios with the S&P 500. Long-only, monthly rebalancing, 252-day estimation window,
10 bps transaction costs. Weights are computed on past data only and applied the next day (no look-ahead;
covered by tests). The benchmark is quoted in USD.

## Project layout

```
main.py                 CLI entry point
app.py                  Streamlit interface
demo.py                 Run the tools on live data, no LLM key needed
src/finance.py          Volatility, return, Sharpe, drawdown, whole-share allocation
src/optimization.py     Min variance, max Sharpe, risk parity (SLSQP, weight caps)
src/strategies.py       Strategies: window of returns -> weights
src/backtest.py         Walk-forward backtest, costs, benchmark comparison
src/tools.py            CrewAI tools (yfinance I/O + JSON for the agents)
src/facts.py            Runs the tools to collect the report's numeric facts
src/report_builder.py   Renders the numeric report sections in Python
src/guardrails.py       Deterministic report validation
src/pipeline.py         Report assembly: numbers + narrative + validation footer
src/agents.py, tasks.py Agents and sequential tasks
src/llm.py              LLM provider configuration
reports/report_pdf.py   Markdown -> PDF
evaluation/             Offline evaluation scenarios and runner
tests/                  pytest suite (no network)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then add your key
```

Get a free Groq key at <https://console.groq.com> and set it in `.env`:

```env
GROQ_API_KEY=your_key
LLM_MODEL=groq/openai/gpt-oss-120b
```

Any [LiteLLM](https://docs.litellm.ai/docs/providers) provider works: set `LLM_MODEL` (for example
`mistral/mistral-small-latest`) and the matching `<PROVIDER>_API_KEY`.

## Usage

```bash
python demo.py                                                   # tools only, no LLM
python main.py --tickers AAPL,MSFT --budget 5000 --profil moderate   # full pipeline
streamlit run app.py                                             # web interface
```

Without an API key the Streamlit app falls back to a local mode (same calculations, no LLM text).
The Groq free tier is limited to 8,000 tokens per minute: keep to 2-3 tickers per run.

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest
```

CI runs the same checks on every push (`.github/workflows/ci.yml`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cookie/crumb fetch failed (SSLError)` | The project or venv path contains accents (for example `é`). Copy `certifi/cacert.pem` to an ASCII-only folder and set `CURL_CA_BUNDLE` in `.env`, or move the venv. |
| `Rate limit reached ... TPM` | Groq free tier: use fewer tickers; automatic retries absorb short peaks. |
| `<PROVIDER>_API_KEY non définie` | Check `.env` and that `LLM_MODEL` matches the key you set. |
| Empty data for a ticker | Check the symbol on Yahoo Finance (`BNP.PA`, `ASML.AS`, `AAPL`). |

*Educational project. Not investment advice.*
