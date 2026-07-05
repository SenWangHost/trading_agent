# Trading Agent

An AI-powered Robinhood portfolio analyzer built with LangGraph and Claude.

## Overview

Fetches your current Robinhood holdings, runs parallel technical / fundamental / news analysis per position via LangGraph, synthesizes signals through Claude, and produces a trading decision report. Trade execution is wired but inactive — the agent reports only until you switch it on.

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # fill in credentials
```

## Usage

```bash
# Run one analysis cycle and exit
uv run python main.py once

# Start the scheduler — runs immediately, then every 30 min Mon–Fri 9:00–15:30 ET
uv run python main.py schedule

# Show help
uv run python main.py --help
```

Reports are printed to stdout and saved to `logs/reports/<timestamp>.md`.  
Raw signal data is appended to `logs/decisions.jsonl`.

## Environment Variables

| Variable | Description |
|---|---|
| `ROBINHOOD_USERNAME` | Robinhood login email |
| `ROBINHOOD_PASSWORD` | Robinhood password |
| `ALPACA_API_KEY` | Alpaca API key (paper trading, reserved for future execution) |
| `ALPACA_SECRET_KEY` | Alpaca secret key |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` for paper |
| `POLYGON_API_KEY` | Polygon.io key for news data |
| `ANTHROPIC_API_KEY` | Claude API key |

## Running Tests

```bash
uv run pytest
uv run pytest tests/test_supervisor.py -v   # single file
```

## Project Structure

```
main.py           — CLI entry point (once / schedule)
scheduler.py      — run_once(), start_scheduler(), cycle logic, report saving
graph.py          — LangGraph StateGraph (fan-out per ticker → supervisor → report)
state.py          — Pydantic models + AgentState TypedDict
agents/
  technical.py    — RSI, MACD, SMA via robin_stocks + pandas-ta
  fundamental.py  — P/E, P/B, market cap via robin_stocks (daily cache)
  news.py         — Headlines via Polygon.io
  supervisor.py   — LLM synthesis → TradeDecision per position
broker/
  base.py         — BaseBroker ABC
  alpaca.py       — AlpacaBroker (active, paper mode)
  robinhood.py    — stub (NotImplementedError)
  ibkr.py         — stub (NotImplementedError)
logs/
  decisions.jsonl — raw signal + decision log (one entry per cycle)
  reports/        — markdown report per cycle
```
