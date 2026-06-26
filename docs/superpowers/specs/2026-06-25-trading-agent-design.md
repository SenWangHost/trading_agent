# Trading Agent — Design Spec

**Date:** 2026-06-25  
**Status:** Draft

---

## Overview

An AI-powered intraday trading agent for US stocks and ETFs. The agent monitors a configurable watchlist, analyzes each ticker across three signal types (technical, fundamental, news/sentiment), and produces paper trade decisions using an LLM-driven supervisor. The system starts with Alpaca paper trading and is designed to migrate to Interactive Brokers (IBKR) without architectural changes.

---

## Architecture

The agent runs as a **LangGraph state machine** triggered on a fixed cadence (every 30 minutes) during market hours (9:30am–4:00pm ET). Each cycle flows through a directed graph:

```
[Scheduler]
    │
    ▼
[Data & Analysis — parallel per ticker]
    ├── Technical Analyzer   → TechnicalSignal
    ├── Fundamental Analyzer → FundamentalSignal  (cached daily)
    └── News/Sentiment Agent → NewsSignal
    │
    ▼
[Supervisor Agent]
    │  (synthesizes all signals via LLM)
    ▼
[Trade Executor]
    │  (broker abstraction layer)
    ├── Alpaca (paper) ← active now
    └── IBKR (paper/live) ← future
    │
    ▼
[Decision Log]  →  logs/decisions.jsonl
```

---

## Components

### Agent Layer (`agents/`)

| File | Responsibility |
|---|---|
| `agents/technical.py` | Fetches intraday OHLCV bars from Polygon.io and historical context from yfinance. Computes RSI, MACD, and moving averages via `pandas-ta`. Returns a `TechnicalSignal`. |
| `agents/fundamental.py` | Pulls P/E, EPS, revenue growth, and debt ratios from Financial Modeling Prep (free tier). Caches output daily — fundamentals do not change intraday. Returns a `FundamentalSignal`. |
| `agents/news.py` | Fetches recent headlines and article summaries for a ticker from Polygon.io news. LLM scores sentiment and flags material events (earnings, guidance, macro). Returns a `NewsSignal`. |
| `agents/supervisor.py` | Receives all three signals. Constructs a synthesis prompt and calls the LLM to produce a `TradeDecision` (action, ticker, size, rationale). |

### Broker Layer (`broker/`)

The broker layer is an **abstract interface** so the execution backend can be swapped without touching agent logic.

| File | Responsibility |
|---|---|
| `broker/base.py` | Abstract base class defining `get_positions()`, `place_order()`, `get_portfolio_value()`. |
| `broker/alpaca.py` | Alpaca paper trading implementation using `alpaca-trade-api`. Active for initial development. |
| `broker/ibkr.py` | IBKR implementation (stubbed initially). Will use IBKR Client Portal API or TWS API once activated. |

### Core (`state.py`, `graph.py`, `scheduler.py`)

| File | Responsibility |
|---|---|
| `state.py` | Defines `AgentState` TypedDict and all Pydantic signal/decision models. |
| `graph.py` | Assembles the LangGraph `StateGraph`, wires nodes and edges, compiles the runnable. |
| `scheduler.py` | Runs the compiled graph on a 30-minute cadence during market hours using APScheduler. |

---

## State & Data Flow

All inter-agent data is typed via Pydantic models validated before it reaches the supervisor. This keeps supervisor reasoning deterministic and auditable.

```python
class TechnicalSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float          # 0.0 – 1.0
    rsi: float
    macd_signal: str
    reasoning: str

class FundamentalSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    pe_ratio: float | None
    revenue_growth_yoy: float | None
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
    size_pct: float            # % of portfolio to allocate
    rationale: str

class AgentState(TypedDict):
    tickers: list[str]
    technical_signals: dict[str, TechnicalSignal]
    fundamental_signals: dict[str, FundamentalSignal]
    news_signals: dict[str, NewsSignal]
    decisions: list[TradeDecision]
    cycle_timestamp: str
```

Each cycle, the three analyzers run in **parallel** per ticker via LangGraph's `Send` API. Results are collected into `AgentState` before the supervisor node runs.

---

## LLM Configuration

| Parameter | Value |
|---|---|
| Model | `claude-sonnet-4-6` |
| Structured output | JSON mode via `model.with_structured_output(SignalModel)` |
| Tracing | LangSmith (`LANGSMITH_API_KEY`) |

Each analyzer and the supervisor use `with_structured_output` to enforce Pydantic model parsing. On parse failure, the agent retries once with a stricter prompt, then emits a neutral/hold signal with an error annotation rather than crashing the cycle.

---

## Error Handling

| Failure mode | Behavior |
|---|---|
| Data API timeout/failure | Agent retries once, then returns a `null` signal with reason. Supervisor treats missing signals as neutral (default: hold). |
| LLM structured output parse failure | Retry once with stricter prompt. After two failures, emit hold decision with `error` flag. |
| Broker API failure | Log error, skip execution for this cycle. Do not halt the scheduler. |

---

## Observability

- **Decision log** — every cycle appends a JSON entry to `logs/decisions.jsonl`:  
  `{ timestamp, ticker, technical_signal, fundamental_signal, news_signal, decision, rationale }`
- **LangSmith tracing** — full LLM call traces for every analyzer and supervisor call. Set `LANGSMITH_API_KEY` in `.env`.
- **Console output** — each cycle prints a summary table of signals and decisions.

---

## Environment Variables

```
ANTHROPIC_API_KEY         # LLM calls
ALPACA_API_KEY            # Alpaca paper trading
ALPACA_SECRET_KEY
ALPACA_BASE_URL           # https://paper-api.alpaca.markets
POLYGON_API_KEY           # Real-time market data + news
FMP_API_KEY               # Financial Modeling Prep (fundamentals)
LANGSMITH_API_KEY         # Optional: LangSmith tracing
LANGSMITH_PROJECT         # Optional: project name in LangSmith
```

---

## Project Structure

```
trading_agent/
├── agents/
│   ├── technical.py
│   ├── fundamental.py
│   ├── news.py
│   └── supervisor.py
├── broker/
│   ├── base.py
│   ├── alpaca.py
│   └── ibkr.py           # stubbed
├── logs/
│   └── decisions.jsonl
├── docs/
│   └── superpowers/specs/
│       └── 2026-06-25-trading-agent-design.md
├── state.py
├── graph.py
├── scheduler.py
├── main.py
├── pyproject.toml         # managed with uv
├── .env.example
├── CLAUDE.md
└── README.md
```

---

## Testing

| Layer | Approach |
|---|---|
| Analyzer agents | Unit tests with mocked API responses. Validates Pydantic model parsing for all signal types. |
| Supervisor | Fixed signal inputs → expected decisions. Uses `langchain_core` fake LLM — no live API calls. |
| Broker implementations | Mock broker calls in unit tests; Alpaca sandbox for integration tests. |
| End-to-end | One full cycle against Alpaca paper API with a single ticker (AAPL) to confirm graph execution. |

---

## Migration Path to IBKR

When ready to switch from Alpaca to IBKR:

1. Implement `broker/ibkr.py` using IBKR Client Portal API (REST) or TWS API (`ib_insync`).
2. Update `broker` instantiation in `graph.py` to use `IBKRBroker`.
3. Set IBKR credentials in `.env`.
4. No changes needed in any agent or supervisor code.
