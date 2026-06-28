# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An AI-powered intraday trading agent for US stocks and ETFs. Uses LangGraph to orchestrate parallel per-ticker analysis (technical, fundamental, news/sentiment), synthesizes signals via LLM, and executes paper trades through Alpaca.

## Setup

```bash
uv sync --extra dev       # install all dependencies
cp .env.example .env      # fill in credentials
uv run python main.py     # start the agent
```

## Running Tests

```bash
uv run pytest             # full suite
uv run pytest tests/test_technical.py -v   # single file
```

## Architecture

```
main.py → scheduler.py → graph.py (LangGraph StateGraph)
                               ├── analyze_ticker (per ticker, parallel via Send)
                               │     ├── agents/technical.py   (robin_stocks + pandas-ta)
                               │     ├── agents/fundamental.py (robin_stocks, cached daily)
                               │     └── agents/news.py        (Polygon.io)
                               ├── agents/supervisor.py        (LLM synthesis → TradeDecision)
                               └── trade_executor              (broker abstraction)
                                     └── broker/alpaca.py      (active — paper trading)
```

- **State flows** through `AgentState` (TypedDict in `state.py`). All signals are Pydantic models validated before reaching the supervisor.
- **Parallel fan-out** uses LangGraph's `Send` API: one `analyze_ticker` node per ticker, results merged via `_merge_dicts` reducer.
- **Broker abstraction**: swap execution backend by changing the `AlpacaBroker` instantiation in `scheduler.py`. `broker/robinhood.py` and `broker/ibkr.py` are stubs that raise `NotImplementedError`.
- **Decision log**: every cycle appends to `logs/decisions.jsonl`.
- **Fundamentals cache**: `agents/fundamental.py` caches per-ticker per-day in a module-level dict — Robinhood fundamentals are fetched once daily.

## Data Sources

| Source | Library | What it provides |
|---|---|---|
| Robinhood | `robin_stocks` | Real-time quotes, OHLCV bars, fundamentals |
| Polygon.io | `requests` | News headlines + article summaries |

## Environment Variables

See `.env.example` for all required keys. Critical ones:
- `ROBINHOOD_USERNAME` / `ROBINHOOD_PASSWORD` — robin_stocks auth at startup
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL` — paper trading
- `POLYGON_API_KEY` — news agent
- `ANTHROPIC_API_KEY` — LLM calls (model: `claude-sonnet-4-6`)
