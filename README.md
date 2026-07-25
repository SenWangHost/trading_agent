# Trading Agent

An AI-powered swing trading agent for US stocks and ETFs. Uses LangGraph to orchestrate parallel per-ticker analysis, synthesizes signals via an LLM (DeepSeek V4 Pro over OpenRouter), and executes trades through Alpaca.

## Setup

```bash
uv sync --extra dev
cp .env.prod .env   # fill in credentials
```

## Usage

```bash
# Run one analysis cycle and exit
uv run python main.py once

# Start the scheduler (runs on two cadences, Mon–Fri ET):
#   Market hours  09:00–15:30 — every 30 minutes
#   After-hours   16:00–20:00 — every 60 minutes
uv run python main.py schedule

uv run python main.py --help
```

## Running Tests

```bash
uv run pytest             # full suite (44 tests)
uv run pytest tests/test_technical.py -v
```

## Deployment

```bash
docker build -t trading-agent:latest .
docker compose up -d      # always-on, restarts on crash
docker compose logs -f    # stream logs
```

### DigitalOcean droplet — pre-flight checklist

Deploy target: an Ubuntu droplet running Docker Compose. This is a scheduled
batch job (no inbound web traffic), so a 1 GB / 1 vCPU Basic droplet is enough.

- [ ] **Droplet ready** — Ubuntu 24.04 LTS, SSH key added, Docker Engine +
      `docker compose` plugin installed.
- [ ] **Code on host** — `git clone` the repo (or `rsync` it up).
- [ ] **Secrets copied manually** — `.env.prod` / `.env` are git-ignored
      (`.env.*`), so they are **not** included by `git clone`. Copy the file to
      the droplet over SSH, then move it into place:
      `scp .env.prod root@DROPLET_IP:/root/trading_agent/.env`
- [ ] **`.env` values filled** — `OPENROUTER_API_KEY`, `POLYGON_API_KEY`,
      `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`, `WATCHLIST`
      (+ scanner vars if used). See the table below.
- [ ] **Paper vs live confirmed** — `ALPACA_BASE_URL=https://paper-api.alpaca.markets`
      for paper; only switch to the live URL once you trust the agent.
- [ ] **Timezone** — no action needed. The scheduler pins `America/New_York` on
      the `BlockingScheduler` and every `CronTrigger`, so cycles fire at correct
      ET times even though the droplet clock is UTC.
- [ ] **Build & run** — `docker build -t trading-agent:latest .` then
      `docker compose up -d` (starts in `schedule` mode, `restart: unless-stopped`).
- [ ] **Verify** — `docker compose ps` shows *running*; `docker compose logs -f`
      shows `technical → done ... price=...`; force a cycle with
      `docker compose exec trading-agent uv run python main.py once`.
- [ ] **Firewall** — `ufw allow OpenSSH && ufw enable` (no app ports needed).

Reports and decisions persist in the named `logs` Docker volume
(`/app/logs/reports`, `/app/logs/decisions`) and survive container restarts.

To update after a code change: `git pull` → rebuild the image → `docker compose up -d`.

## Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key — all LLM calls |
| `LLM_MODEL` | OpenRouter model id (default `deepseek/deepseek-v4-pro`) |
| `POLYGON_API_KEY` | Polygon.io — all market data (bars, fundamentals, news, scanner) |
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_SECRET_KEY` | Alpaca secret key |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` for paper, live URL for real trades |
| `WATCHLIST` | Comma-separated tickers, e.g. `AAPL,MSFT,NVDA` |
| `SCANNER_ENABLED` | `true` to auto-expand watchlist via Polygon scanner (default: `false`) |
| `SCANNER_SECTORS` | Comma-separated sector names, e.g. `semiconductors,technology` |
| `SCANNER_MAX_RESULTS` | Max tickers added by scanner (default: `10`) |
| `SCANNER_MIN_PRICE` | Exclude stocks below this price (default: `10`) |
| `SCANNER_MIN_VOLUME` | Exclude stocks below this daily volume (default: `1000000`) |
| `LANGCHAIN_API_KEY` | Optional — LangSmith tracing |

Available sectors: `technology`, `healthcare`, `financials`, `energy`, `consumer_discretionary`, `consumer_staples`, `industrials`, `materials`, `real_estate`, `utilities`, `communication_services`, `semiconductors`

## Project Structure

```
main.py               — CLI entry point (once | schedule)
scheduler.py          — cycle runner, position/order fetching, report saving
graph.py              — LangGraph StateGraph (fan-out → supervisor → execute → report)
state.py              — Pydantic models + AgentState TypedDict
polygon_client.py     — shared Polygon HTTP client with retry-on-429 backoff
agents/
  technical.py        — 90-day daily bars (Polygon) + pandas-ta indicators;
                        near-real-time price via latest Polygon 1-min bar
                        (incl. pre/after-hours; ~15-min delayed on Starter tier)
  fundamental.py      — Polygon financials + yfinance earnings calendar (daily cache)
  news.py             — Polygon news headlines + LLM sentiment scoring
  supervisor.py       — synthesizes signals + recent orders → TradeDecision
broker/
  base.py             — BaseBroker ABC
  alpaca.py           — AlpacaBroker: market + limit orders, portfolio value
  robinhood.py        — stub
  ibkr.py             — stub
scanner/
  polygon.py          — grouped daily bars → top-N candidates by volume
  sectors.py          — GICS sector universe + aliases
logs/
  reports/<date>/     — markdown report per cycle  (e.g. 09-30-00.md)
  decisions/<date>/   — decisions.json (JSON array, one entry per cycle)
tests/                — 44 tests across all layers
Dockerfile
docker-compose.yml
k8s/                  — Kubernetes manifests (namespace, secret, configmap, pvc, cronjob)
```
