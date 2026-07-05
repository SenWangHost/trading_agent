# Trading Agent — Design Spec

**Date:** 2026-06-25  
**Updated:** 2026-07-04  
**Status:** Active

---

## Overview

An AI-powered portfolio analysis agent for US stocks and ETFs. The agent fetches the current Alpaca account positions (or a configurable watchlist when no positions are held), analyzes each ticker across three signal dimensions (technical, fundamental, news/sentiment), synthesizes signals via Claude, and produces a **trading decision report**. Trade execution is wired but inactive — the broker layer is kept intact so execution can be enabled without architectural changes.

---

## Architecture

The agent has two clearly separated concerns:

- **Data layer** — Polygon.io only (one API key, REST via `requests`)
- **Broker layer** — Alpaca (active), with IBKR and Robinhood stubs ready to swap in

```
[CLI: main.py once | schedule]
    │
    ▼
[Scheduler: fetch broker positions + WATCHLIST tickers]
    │
    ▼
[Data & Analysis — parallel per ticker via LangGraph Send]
    ├── Technical Analyzer   ─── Polygon /v2/aggs bars ──► TechnicalSignal
    ├── Fundamental Analyzer ─── Polygon /vX/reference/financials ──► FundamentalSignal  (cached daily)
    └── News/Sentiment Agent ─── Polygon /v2/reference/news ──► NewsSignal
    │
    ▼
[Supervisor Agent]
    │  (synthesizes all signals + portfolio position context via LLM)
    ▼
[Report Generator]           ← active
    │  (markdown report per cycle)
    ▼
[Decision Log]  →  logs/decisions.jsonl
                →  logs/reports/<timestamp>.md

── [Trade Executor] ──────────── inactive (kept for future activation)
    ├── Alpaca paper/live ← ready
    ├── IBKR ← stub
    └── Robinhood ← stub
```

---

## Components

### Agent Layer (`agents/`)

| File | Responsibility |
|---|---|
| `agents/technical.py` | Fetches 5-min OHLCV bars from Polygon (`/v2/aggs/ticker/{ticker}/range/5/minute`). Computes RSI, MACD, and moving averages via `pandas-ta`. Returns a `TechnicalSignal`. |
| `agents/fundamental.py` | Fetches ticker details and trailing twelve months financials from Polygon.io (`/v3/reference/tickers`, `/vX/reference/financials`). Caches output daily. Returns a `FundamentalSignal`. |
| `agents/news.py` | Fetches recent headlines and article summaries from Polygon (`/v2/reference/news`). LLM scores sentiment and flags material events. Returns a `NewsSignal`. |
| `agents/supervisor.py` | Receives all three signals plus the current portfolio position (shares held, avg cost, unrealized P&L). Constructs a synthesis prompt and calls Claude to produce a `TradeDecision`. |

### Broker Layer (`broker/`)

The broker layer is an **abstract interface** so the execution backend can be swapped without touching agent logic. Currently inactive in the graph but fully implemented.

| File | Responsibility |
|---|---|
| `broker/base.py` | Abstract base class defining `get_positions()`, `place_order()`, `get_portfolio_value()`. |
| `broker/alpaca.py` | Alpaca paper trading implementation using `alpaca-py`. Ready for activation. |
| `broker/robinhood.py` | Stub — raises `NotImplementedError`. |
| `broker/ibkr.py` | Stub — raises `NotImplementedError`. |

### Core (`state.py`, `graph.py`, `scheduler.py`)

| File | Responsibility |
|---|---|
| `state.py` | Defines `AgentState` TypedDict, `PortfolioPosition`, and all Pydantic signal/decision models. |
| `graph.py` | Assembles the LangGraph `StateGraph`. Active terminal node is `report_generator`. `execute_trades` function is kept intact — swap the node registration to enable live execution. |
| `scheduler.py` | Fetches Alpaca positions each cycle, reads `WATCHLIST` for tickers, runs the compiled graph, saves report to `logs/reports/`. |
| `main.py` | CLI entry point with `argparse`. Two subcommands: `once` (single cycle, then exit) and `schedule` (APScheduler loop). No login required. |

---

## State & Data Flow

All inter-agent data is typed via Pydantic models validated before it reaches the supervisor.

```python
class PortfolioPosition(BaseModel):
    ticker: str
    quantity: float
    average_buy_price: float
    current_value: float
    equity_change_pct: float      # unrealized P&L %

class TechnicalSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float             # 0.0 – 1.0
    rsi: float
    macd_signal: str
    current_price: float
    reasoning: str

class FundamentalSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    pe_ratio: float | None
    pb_ratio: float | None
    market_cap: float | None
    week_52_high: float | None
    week_52_low: float | None
    reasoning: str

class NewsSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    material_event: bool
    reasoning: str

class TradeDecision(BaseModel):
    ticker: str
    action: Literal["buy", "sell", "hold"]
    size_pct: float               # % of portfolio to add or reduce
    rationale: str

class AgentState(TypedDict):
    tickers: list[str]
    current_ticker: str
    portfolio_positions: dict[str, PortfolioPosition]
    prices: dict[str, float]
    technical_signals: dict[str, TechnicalSignal]
    fundamental_signals: dict[str, FundamentalSignal]
    news_signals: dict[str, NewsSignal]
    decisions: list[TradeDecision]
    cycle_timestamp: str
    report: str
```

Each cycle, the three analyzers run in **parallel** per ticker via LangGraph's `Send` API. Results are merged into `AgentState` before the supervisor node runs.

---

## LLM Configuration

| Parameter | Value |
|---|---|
| Model | `claude-sonnet-4-6` |
| Structured output | `model.with_structured_output(PydanticModel)` on every LLM call |
| LLM instantiation | Inside each agent function (not at module level) — required for `unittest.mock.patch` to work correctly in tests |
| Tracing | LangSmith (`LANGCHAIN_API_KEY`) — optional |

### Supervisor Prompt

The supervisor prompt includes portfolio position context for each ticker — shares held, average buy price, current value, and unrealized P&L. This allows Claude to give recommendations grounded in the real portfolio state (e.g., protect gains when signals turn bearish, limit further losses on losing positions).

---

## Error Handling

| Failure mode | Behavior |
|---|---|
| Data API timeout/failure | Agent catches exception, returns a neutral signal with error annotation. Supervisor treats missing signals as neutral (default: hold). |
| LLM structured output failure | Caught per-ticker; supervisor emits a hold decision with zero size. |
| Alpaca positions fetch failure | Gracefully returns empty dict; tickers still come from `WATCHLIST`. |
| Broker API failure (execution) | Logged and skipped per-order. Scheduler continues. |

---

## Observability

- **Decision log** — every cycle appends a JSON entry to `logs/decisions.jsonl`:  
  `{ timestamp, tickers, portfolio_positions, technical_signals, fundamental_signals, news_signals, decisions }`
- **Report file** — every cycle writes a markdown report to `logs/reports/<timestamp>.md` with P&L, signals, and recommendations per position
- **Console output** — report is printed to stdout each cycle
- **LangSmith tracing** — set `LANGCHAIN_API_KEY` in `.env` to enable full LLM call traces

---

## Data Sources

### Data (Polygon.io only)

| Endpoint | Provides | Used by |
|---|---|---|
| `GET /v2/aggs/ticker/{t}/range/5/minute/{from}/{to}` | 5-min OHLCV bars (7-day window) | `agents/technical.py` |
| `GET /v3/reference/tickers/{ticker}` | Market cap, shares outstanding, description | `agents/fundamental.py` |
| `GET /vX/reference/financials` | Quarterly income statement + balance sheet (TTM) | `agents/fundamental.py` |
| `GET /v2/reference/news` | News headlines + summaries (past 7 days) | `agents/news.py` |

All data calls use a single `POLYGON_API_KEY`. Free tier gives 15-min delayed data, which is acceptable for report mode.

> Polygon fundamentals do not include current price, so `pe_ratio` and `pb_ratio` in `FundamentalSignal` are always `null`. The LLM synthesizes direction from revenue trend, net income, EPS, and balance sheet metrics instead.

### Broker (Alpaca — swappable)

| Library | Provides | Used by |
|---|---|---|
| `alpaca-py` `TradingClient` | Current positions, account value, order execution | `broker/alpaca.py`, `scheduler.py` |

---

## CLI Usage

```bash
# Single analysis cycle — fetches portfolio, generates report, exits
uv run python main.py once

# Scheduled mode — runs immediately, then every 30 min Mon–Fri 9:00–15:30 ET
uv run python main.py schedule

# Help
uv run python main.py --help
```

---

## Environment Variables

```
ANTHROPIC_API_KEY         # LLM calls (Claude)
ALPACA_API_KEY            # Market data, news, positions, and paper trading
ALPACA_SECRET_KEY
ALPACA_BASE_URL           # https://paper-api.alpaca.markets
POLYGON_API_KEY           # No longer used — kept in .env.example for reference only
LANGCHAIN_API_KEY         # Optional: LangSmith tracing
LANGCHAIN_PROJECT         # Optional: project name in LangSmith
WATCHLIST                 # Comma-separated tickers, e.g. AAPL,MSFT,NVDA
```

---

## Enabling Live Trade Execution

The broker layer and `execute_trades` function in `graph.py` are fully implemented and tested. To switch from report mode to live execution, make two edits in `graph.py`:

```python
# Replace:
builder.add_node("report_generator", generate_report)
builder.add_edge("supervisor", "report_generator")
builder.add_edge("report_generator", END)

# With:
builder.add_node("trade_executor", execute_trades)
builder.add_edge("supervisor", "trade_executor")
builder.add_edge("trade_executor", END)
```

No changes needed in any agent, supervisor, or scheduler code.

> **Warning:** Activating `trade_executor` with `ALPACA_BASE_URL=https://api.alpaca.markets` (not the paper URL) will place real orders.

---

## Project Structure

```
trading_agent/
├── agents/
│   ├── technical.py        # alpaca-py bars + pandas-ta
│   ├── fundamental.py      # yfinance (daily cache)
│   ├── news.py             # alpaca-py NewsClient
│   └── supervisor.py       # LLM synthesis with position context
├── broker/
│   ├── base.py             # BaseBroker ABC
│   ├── alpaca.py           # AlpacaBroker (active, paper mode)
│   ├── robinhood.py        # stub
│   └── ibkr.py             # stub
├── logs/
│   ├── decisions.jsonl     # raw signal + decision log
│   └── reports/            # markdown report per cycle
├── tests/
│   ├── test_state.py
│   ├── test_broker.py
│   ├── test_technical.py
│   ├── test_fundamental.py
│   ├── test_news.py
│   ├── test_supervisor.py
│   └── test_graph.py
├── docs/superpowers/
│   ├── specs/2026-06-25-trading-agent-design.md  ← this file
│   └── plans/2026-06-28-trading-agent.md
├── state.py
├── graph.py
├── scheduler.py
├── main.py
├── pyproject.toml          # uv project (Python 3.12+)
├── .env.example
├── CLAUDE.md
└── README.md
```

---

## Testing

```bash
uv run pytest             # full suite (27 tests)
uv run pytest -v          # verbose
```

| Layer | Approach |
|---|---|
| Analyzer agents | Unit tests with mocked Alpaca/yfinance responses and mocked LLM |
| Supervisor | Fixed signal inputs + mocked LLM; asserts position context appears in prompt |
| Broker | Mock `TradingClient`; verify stubs raise `NotImplementedError` |
| Graph | Full cycle with stub broker and all agents mocked; asserts report generated |

---

## Execution Phase Roadmap

| Phase | Status | Notes |
|---|---|---|
| Report mode (current) | Active | Analyzes watchlist, generates report, no orders placed |
| Paper trading | Ready (not activated) | Swap `report_generator` → `trade_executor` in `graph.py`; set `ALPACA_BASE_URL` to paper URL |
| Live trading — Alpaca | Future | Change paper URL to live URL; validate strategy thoroughly first |
| Live trading — IBKR | Future | Implement `broker/ibkr.py`; swap broker in `scheduler.py` |
