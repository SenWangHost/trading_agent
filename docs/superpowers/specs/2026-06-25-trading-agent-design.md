# Trading Agent — Design Spec

**Date:** 2026-06-25  
**Updated:** 2026-07-07  
**Status:** Active — paper trading live on Alpaca

---

## Overview

An AI-powered swing trading agent for US stocks and ETFs. Each cycle the agent:

1. Fetches current Alpaca positions and recent orders (last 7 days)
2. Builds a watchlist (static `WATCHLIST` + optional Polygon scanner candidates)
3. Runs parallel per-ticker analysis (technical, fundamental, news/sentiment)
4. Synthesizes signals via Claude into a `TradeDecision` with explicit order type and limit price
5. Executes trades via Alpaca (market or limit orders)
6. Saves a markdown report and JSON decision log

---

## Architecture

```
main.py  (argparse: once | schedule)
  └── scheduler.py
        ├── _fetch_alpaca_positions()       Alpaca TradingClient → current holdings
        ├── _fetch_recent_orders()          Alpaca TradingClient → last 7 days orders
        ├── _build_watchlist()              WATCHLIST env var + optional scanner
        │     └── scanner/polygon.py        Polygon grouped daily bars → top-N by volume
        │           └── scanner/sectors.py  GICS sector universe + aliases
        └── graph.py  (LangGraph StateGraph)
              ├── dispatch                  fan-out: one analyze_ticker node per ticker
              ├── analyze_ticker            (parallel via LangGraph Send API)
              │     ├── agents/technical.py    Polygon daily bars + yfinance price
              │     ├── agents/fundamental.py  Polygon financials + yfinance earnings
              │     └── agents/news.py         Polygon news headlines
              ├── agents/supervisor.py      LLM synthesis → TradeDecision
              ├── execute_trades            Alpaca market/limit orders
              └── report_generator          markdown report + JSON decision log
```

---

## Components

### Data Layer

All market data comes from **Polygon.io** via a shared `polygon_client.py` that handles retry-on-429 with exponential backoff (15 s, 30 s, 45 s + jitter). Real-time current price is fetched separately via **yfinance** `fast_info.last_price` (no API key required).

| Source | Endpoint / Method | Provides | Used by |
|---|---|---|---|
| Polygon | `/v2/aggs/grouped/locale/us/market/stocks/{date}` | All US stocks OHLCV — scanner | `scanner/polygon.py` |
| Polygon | `/v2/aggs/ticker/{t}/range/1/day` | 90-day daily bars | `agents/technical.py` |
| Polygon | `/vX/reference/financials` | Quarterly income statement + balance sheet (TTM) | `agents/fundamental.py` |
| Polygon | `/v3/reference/tickers/{t}` | Market cap, shares outstanding, description | `agents/fundamental.py` |
| Polygon | `/v2/reference/news` | News headlines + summaries (past 7 days) | `agents/news.py` |
| yfinance | `Ticker.fast_info.last_price` | Real-time price (regular hours) or last close | `agents/technical.py` |
| yfinance | `Ticker.calendar` | Next earnings date + EPS/revenue estimates | `agents/fundamental.py` |
| Alpaca | `TradingClient.get_all_positions()` | Current holdings | `scheduler.py` |
| Alpaca | `TradingClient.get_orders()` | Orders from last 7 days (all statuses) | `scheduler.py` |
| Alpaca | `TradingClient.submit_order()` | Order execution | `broker/alpaca.py` |

### Agent Layer (`agents/`)

**`technical.py`** — Fetches 90-day daily OHLCV bars from Polygon. Computes RSI(14), MACD(12/26/9), SMA(20/50), ATR(14), and volume ratio vs 20-day average via `pandas-ta`. Fetches the current price via yfinance (`fast_info.last_price`), which returns the real-time intraday price during market hours and the official session close otherwise. Returns a `TechnicalSignal`.

**`fundamental.py`** — Fetches quarterly financials from Polygon to compute TTM EPS, revenue, and P/E ratio. Fetches market cap and shares outstanding. Augments with next earnings date and estimates from yfinance. Caches output per-ticker per-day to avoid redundant API calls. Returns a `FundamentalSignal`.

**`news.py`** — Fetches the last 7 days of news headlines and summaries from Polygon. Passes them to Claude to score sentiment direction and flag material events (earnings surprises, FDA decisions, mergers). Returns a `NewsSignal`.

**`supervisor.py`** — Receives all three signals plus current portfolio position and the ticker's recent orders (last 7 days). Constructs a synthesis prompt and calls Claude with structured output to produce a `TradeDecision` including action, portfolio size %, order type, and limit price.

### Broker Layer (`broker/`)

Abstract interface so the execution backend is swappable without touching agent logic.

| File | Responsibility |
|---|---|
| `broker/base.py` | ABC defining `get_positions()`, `place_order(ticker, action, qty, order_type, limit_price)`, `get_portfolio_value()` |
| `broker/alpaca.py` | Routes to `MarketOrderRequest` or `LimitOrderRequest` based on `order_type`; rounds limit price to 2 dp |
| `broker/robinhood.py` | Stub — raises `NotImplementedError` |
| `broker/ibkr.py` | Stub — raises `NotImplementedError` |

### Scanner (`scanner/`)

When `SCANNER_ENABLED=true`, fetches Polygon's grouped daily bars endpoint (all US stocks in one call) and filters by price, volume, and optional sector universe. Returns the top-N tickers by volume to augment the static `WATCHLIST`.

`scanner/sectors.py` defines a curated GICS universe (12 sectors + `semiconductors` sub-sector) with aliases (`semi` → `semiconductors`, `tech` → `technology`, etc.).

---

## State & Data Flow

```python
class RecentOrder(BaseModel):
    order_id: str; ticker: str; action: str; order_type: str
    qty: float; filled_qty: float; limit_price: float | None
    status: str; submitted_at: str; filled_at: str | None

class TradeDecision(BaseModel):
    ticker: str
    action: Literal["buy", "sell", "hold"]
    size_pct: float                      # % of total portfolio to deploy/reduce
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None     # set when order_type == "limit"
    rationale: str

class AgentState(TypedDict):
    tickers: list[str]
    current_ticker: str
    portfolio_positions: dict[str, PortfolioPosition]
    recent_orders: list[RecentOrder]
    prices: dict[str, float]            # live prices (from yfinance)
    technical_signals: dict[str, TechnicalSignal]
    fundamental_signals: dict[str, FundamentalSignal]
    news_signals: dict[str, NewsSignal]
    decisions: list[TradeDecision]
    cycle_timestamp: str
    report: str
```

Per-ticker parallel fan-out uses LangGraph's `Send` API. Results merge into `AgentState` before the supervisor node runs.

---

## Order Execution

Trades are active. The supervisor prompt instructs Claude to:

- **Prefer limit orders** for swing trading (most buys/sells) — limit price set 0.2–0.5% from current price or at a technical support/resistance level
- **Use market orders** only when urgency outweighs price precision (material news event, strong breakout, urgent stop-loss exit)
- Skip buy if the ticker is already held (no doubling up)
- Sell using the exact held quantity

Execution guards in `graph.py`:
```python
if decision.action == "buy" and held:
    skip  # already holding — swing trading no-double-up rule
if decision.action == "sell" and not held:
    skip  # nothing to sell
qty = held.quantity  # for sells, always use exact held quantity
```

---

## Scheduling

`start_scheduler()` uses APScheduler `BlockingScheduler` with two `CronTrigger` jobs (Mon–Fri ET):

| Window | Cadence |
|---|---|
| Market hours 09:00–15:30 | Every 30 minutes |
| After-hours 16:00–20:00 | Every 60 minutes |

On startup, the agent checks the current ET time and skips the initial cycle if it is outside both windows (weekends, holidays, overnight).

---

## Observability

**Logs** — Python `logging` module throughout; `basicConfig` in `main.py` with format `%(asctime)s [%(levelname)s] %(name)s — %(message)s`.

**Decision log** — every cycle appends one entry to `logs/decisions/<YYYY-MM-DD>/decisions.json` (a JSON array, one object per cycle):
```json
{ "timestamp", "tickers", "portfolio_positions", "recent_orders",
  "technical_signals", "fundamental_signals", "news_signals", "decisions" }
```

**Report file** — every cycle writes `logs/reports/<YYYY-MM-DD>/<HH-MM-SS>.md` with per-ticker position, signals, recent orders, and recommendation including order type and limit price.

**LangSmith tracing** — set `LANGCHAIN_API_KEY` in `.env` to enable full LLM call traces.

---

## Deployment

### Local (docker-compose)

```bash
docker build -t trading-agent:latest .
docker compose up -d      # restart: unless-stopped
```

Logs are persisted in a named Docker volume (`logs`). To use a bind mount instead, change `docker-compose.yml` to `./logs:/app/logs`.

### Cloud VM (DigitalOcean Droplet or equivalent)

Recommended: Basic 1 vCPU / 1 GB RAM droplet (~$6/month), Ubuntu 24.04 LTS.

```bash
ssh root@<ip>
curl -fsSL https://get.docker.com | sh
git clone https://github.com/SenWangHost/trading_agent.git && cd trading_agent
scp .env.prod root@<ip>:/root/trading_agent/.env
docker build -t trading-agent:latest .
docker compose up -d
```

### Kubernetes (optional)

`k8s/` contains manifests for a CronJob deployment. The CronJob runs `main.py once` on a schedule and is appropriate if you prefer Kubernetes-managed scheduling over the APScheduler approach.

---

## Environment Variables

```
ANTHROPIC_API_KEY         LLM calls (claude-sonnet-4-6)
POLYGON_API_KEY           All market data: bars, financials, news, scanner
ALPACA_API_KEY            Broker: positions, orders, execution
ALPACA_SECRET_KEY
ALPACA_BASE_URL           https://paper-api.alpaca.markets (paper) | live URL
WATCHLIST                 Comma-separated tickers, e.g. AAPL,MSFT,NVDA
SCANNER_ENABLED           true | false (default: false)
SCANNER_SECTORS           e.g. semiconductors,technology (blank = all sectors)
SCANNER_MAX_RESULTS       Max tickers from scanner (default: 10)
SCANNER_MIN_PRICE         Minimum stock price filter (default: 10)
SCANNER_MIN_VOLUME        Minimum daily volume filter (default: 1000000)
LANGCHAIN_API_KEY         Optional: LangSmith tracing
LANGCHAIN_PROJECT         Optional: LangSmith project name
```

---

## Testing

```bash
uv run pytest             # 44 tests
uv run pytest -v          # verbose
```

| Layer | Approach |
|---|---|
| `test_technical.py` | Mocked Polygon bars + yfinance price; mocked LLM |
| `test_fundamental.py` | Mocked Polygon financials + yfinance earnings; mocked LLM |
| `test_news.py` | Mocked Polygon news; mocked LLM |
| `test_supervisor.py` | Fixed signal inputs + mocked LLM; asserts position and order context in prompt |
| `test_graph.py` | Full cycle with stub broker and all agents mocked; asserts report and swing guards |
| `test_scanner.py` | Mocked Polygon grouped bars; sector filtering and alias resolution |
| `test_state.py` | Pydantic model validation |

---

## Project Structure

```
trading_agent/
├── agents/
│   ├── technical.py        Polygon daily bars + yfinance price + pandas-ta
│   ├── fundamental.py      Polygon financials + yfinance earnings (daily cache)
│   ├── news.py             Polygon news + LLM sentiment
│   └── supervisor.py       LLM synthesis → TradeDecision (order type + limit price)
├── broker/
│   ├── base.py             BaseBroker ABC
│   ├── alpaca.py           MarketOrderRequest / LimitOrderRequest
│   ├── robinhood.py        stub
│   └── ibkr.py             stub
├── scanner/
│   ├── polygon.py          grouped daily bars → top-N candidates
│   └── sectors.py          GICS sector universe + aliases
├── logs/
│   ├── reports/<date>/     <HH-MM-SS>.md per cycle
│   └── decisions/<date>/   decisions.json (JSON array)
├── tests/                  44 tests
├── k8s/                    Kubernetes manifests
├── docs/superpowers/
│   ├── specs/              this file
│   └── plans/
├── state.py
├── graph.py
├── scheduler.py
├── main.py
├── polygon_client.py       shared Polygon HTTP client with retry-on-429
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.prod               credential template (gitignored)
└── CLAUDE.md
```
