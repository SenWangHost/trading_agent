# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An AI-powered swing trading agent for US stocks and ETFs. Uses LangGraph to orchestrate parallel per-ticker analysis (technical, fundamental, news/sentiment), synthesizes signals via an LLM (DeepSeek V4 Pro over OpenRouter), executes trades via Alpaca paper/live broker, and generates a markdown report each cycle. A Polygon-based stock scanner automatically expands the watchlist beyond the static list.

## Setup

```bash
uv sync --extra dev       # install all dependencies
cp .env.prod .env         # fill in credentials
uv run python main.py once      # single analysis cycle
uv run python main.py schedule  # run daily at 9:30 AM ET (Mon–Fri)
```

## Running Tests

```bash
uv run pytest             # full suite (34 tests)
uv run pytest tests/test_technical.py -v   # single file
```

## Architecture

Three clearly separated concerns:

- **Scanner** — Polygon grouped daily bars → filtered candidate list (optional, opt-in)
- **Data** — Polygon.io only (`requests`, one key for all per-ticker data)
- **Broker** — swappable abstraction (Alpaca active, IBKR/Robinhood stubs)

```
main.py (argparse: once | schedule)
  └── scheduler.py
        ├── scanner/polygon.py       (optional: Polygon grouped bars → top-N candidates)
        ├── fetch positions from broker (Alpaca TradingClient)
        └── graph.py (LangGraph StateGraph)
              ├── analyze_ticker (per ticker, parallel via Send)
              │     ├── agents/technical.py   (Polygon daily bars + pandas-ta: RSI/MACD/SMA/ATR/volume)
              │     ├── agents/fundamental.py (Polygon financials + yfinance earnings, cached daily)
              │     └── agents/news.py        (Polygon news)
              ├── agents/supervisor.py        (LLM synthesis + position context → TradeDecision)
              ├── execute_trades              (active — places market orders via broker)
              └── report_generator            (active — prints + saves markdown report)
                        └── broker/alpaca.py
```

- **State** flows through `AgentState` (TypedDict in `state.py`). All signals are Pydantic models validated before reaching the supervisor.
- **Parallel fan-out** uses LangGraph's `Send` API: one `analyze_ticker` node per ticker.
- **Swing trading guards** in `execute_trades`: skip buy if position already held (no doubling up); sell uses exact held quantity.
- **Broker abstraction**: swap execution backend by changing the `AlpacaBroker` instantiation in `scheduler.py`. `broker/robinhood.py` and `broker/ibkr.py` are stubs.
- **Report output**: every cycle prints to stdout, appends to `logs/decisions/<date>/decisions.jsonl`, and saves to `logs/reports/<date>/<time>.md`.
- **Fundamentals cache**: `agents/fundamental.py` caches per-ticker per-day — Polygon financial statements fetched once daily.

## Data Sources

All market data comes from a single provider (Polygon.io). Broker operations use Alpaca.

| Source | What it provides |
|---|---|
| Polygon `/v2/aggs/grouped/locale/us/market/stocks/{date}` | Scanner: all US stocks' daily OHLCV — filter + rank by volume |
| Polygon `/v2/aggs/ticker/{t}/range/1/day` | 90-day daily bars → RSI, MACD, SMA20/50, ATR(14), volume ratio |
| Polygon `/vX/reference/financials` | Quarterly income statement + balance sheet → TTM EPS, revenue, equity |
| Polygon `/v3/reference/tickers/{t}` | Market cap, shares outstanding, company description |
| Polygon `/v2/reference/news` | News headlines + summaries (past 7 days) |
| yfinance `Ticker.calendar` | Next earnings date + EPS/revenue estimates (no API key required) |
| Alpaca `TradingClient` | Current positions, account value, order execution (broker only) |

## Environment Variables

See `.env.prod` for all required keys. Critical ones:
- `POLYGON_API_KEY` — all market data (bars, fundamentals, news, scanner)
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL` — broker (positions + trading)
- `OPENROUTER_API_KEY` — all LLM calls (OpenAI-compatible, routed through OpenRouter)
- `LLM_MODEL` — OpenRouter model id (default: `deepseek/deepseek-v4-pro`, set in `llm.py`)
- `SCANNER_ENABLED` — set `true` to auto-expand watchlist (default: `false`)
- `SCANNER_MAX_RESULTS` — max tickers added by scanner (default: `10`)
- `SCANNER_MIN_PRICE` — exclude stocks below this price (default: `10`)
- `SCANNER_MIN_VOLUME` — exclude stocks below this daily volume (default: `1000000`)
