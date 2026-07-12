import datetime
import json
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from broker.alpaca import AlpacaBroker
from graph import build_graph
from scanner.polygon import scan_candidates
from state import AgentState, PortfolioPosition, RecentOrder

log = logging.getLogger(__name__)


def _fetch_alpaca_positions() -> dict[str, PortfolioPosition]:
    client = TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )
    positions: dict[str, PortfolioPosition] = {}
    try:
        for p in client.get_all_positions():
            try:
                positions[p.symbol] = PortfolioPosition(
                    ticker=p.symbol,
                    quantity=float(p.qty),
                    average_buy_price=float(p.avg_entry_price),
                    current_value=float(p.market_value),
                    equity_change_pct=float(p.unrealized_plpc) * 100,
                )
            except (ValueError, TypeError):
                pass
    except Exception:
        pass
    return positions


def _fetch_recent_orders(days: int = 7) -> list[RecentOrder]:
    client = TradingClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )
    after = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    orders: list[RecentOrder] = []
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=after, limit=500)
        for o in client.get_orders(filter=req):
            try:
                orders.append(RecentOrder(
                    order_id=str(o.id),
                    ticker=o.symbol,
                    action="buy" if str(o.side).lower().endswith("buy") else "sell",
                    order_type=str(o.order_type).split(".")[-1].lower(),
                    qty=float(o.qty or 0),
                    filled_qty=float(o.filled_qty or 0),
                    limit_price=float(o.limit_price) if o.limit_price else None,
                    status=str(o.status).split(".")[-1].lower(),
                    submitted_at=str(o.submitted_at),
                    filled_at=str(o.filled_at) if o.filled_at else None,
                ))
            except (ValueError, TypeError):
                pass
    except Exception as e:
        log.warning("failed to fetch recent orders: %s", e)
    log.info("fetched %d recent orders (last %d days)", len(orders), days)
    return orders


def _build_watchlist() -> list[str]:
    manual = [t.strip() for t in os.environ.get("WATCHLIST", "AAPL,MSFT,NVDA").split(",")]
    if os.environ.get("SCANNER_ENABLED", "false").lower() != "true":
        return manual

    try:
        scanned = scan_candidates()
    except Exception as e:
        log.warning("scanner failed, using manual watchlist only: %s", e)
        return manual

    seen = set(manual)
    extra = [t for t in scanned if t not in seen]
    if extra:
        log.info("scanner added %d tickers: %s", len(extra), ", ".join(extra))
    return manual + extra


def run_cycle(graph) -> None:
    # Stamp cycles in ET so report/decision filenames line up with the ET
    # trading schedule (a Fri 20:00 ET run should file under Fri, not Sat UTC).
    now = datetime.datetime.now(tz=ZoneInfo("America/New_York"))

    watchlist = _build_watchlist()
    portfolio_positions = _fetch_alpaca_positions()
    recent_orders = _fetch_recent_orders()

    held = [t for t in watchlist if t in portfolio_positions]
    log.info("cycle start | %d tickers: %s", len(watchlist), ", ".join(watchlist))
    if held:
        log.info("positions held in Alpaca: %s", ", ".join(held))
    else:
        log.info("no positions currently held")

    initial_state: AgentState = {
        "tickers": watchlist,
        "current_ticker": "",
        "portfolio_positions": portfolio_positions,
        "recent_orders": recent_orders,
        "prices": {},
        "technical_signals": {},
        "fundamental_signals": {},
        "news_signals": {},
        "decisions": [],
        "cycle_timestamp": now.isoformat(),
        "report": "",
    }

    result = graph.invoke(initial_state)
    _log(result)
    _save_report(result)
    log.info("cycle done | decisions: %s",
             ", ".join(f"{d.ticker}={d.action}" for d in result["decisions"]))


def _day_dir(base: str, date_str: str) -> Path:
    """Return and create logs/<base>/<YYYY-MM-DD>/."""
    path = Path(f"logs/{base}/{date_str}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log(state: AgentState) -> None:
    date_str = state["cycle_timestamp"][:10]          # YYYY-MM-DD
    log_path = _day_dir("decisions", date_str) / "decisions.json"
    entry = {
        "timestamp": state["cycle_timestamp"],
        "tickers": state["tickers"],
        "portfolio_positions": {k: v.model_dump() for k, v in state["portfolio_positions"].items()},
        "recent_orders": [o.model_dump() for o in state["recent_orders"]],
        "technical_signals": {k: v.model_dump() for k, v in state["technical_signals"].items()},
        "fundamental_signals": {k: v.model_dump() for k, v in state["fundamental_signals"].items()},
        "news_signals": {k: v.model_dump() for k, v in state["news_signals"].items()},
        "decisions": [d.model_dump() for d in state["decisions"]],
    }
    if log_path.exists():
        cycles = json.loads(log_path.read_text())
    else:
        cycles = []
    cycles.append(entry)
    log_path.write_text(json.dumps(cycles, indent=2))


def _save_report(state: AgentState) -> None:
    report = state.get("report", "")
    if not report:
        return

    print(report)

    # e.g. "2026-07-04T09-30-00" → date_str="2026-07-04", time_str="09-30-00"
    ts_safe = state["cycle_timestamp"].replace(":", "-")[:19]
    date_str = ts_safe[:10]
    time_str = ts_safe[11:]
    report_path = _day_dir("reports", date_str) / f"{time_str}.md"
    report_path.write_text(report)
    log.info("report saved → %s", report_path)


def _build_graph():
    broker = AlpacaBroker(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )
    return build_graph(broker)


def _in_trading_window() -> bool:
    """Return True if current ET time falls in market or after-hours windows."""
    now = datetime.datetime.now(tz=ZoneInfo("America/New_York"))
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    t = now.time()
    market = datetime.time(9, 0) <= t <= datetime.time(15, 30)
    afterhrs = datetime.time(16, 0) <= t <= datetime.time(20, 0)
    return market or afterhrs


def run_once() -> None:
    """Fetch portfolio, run one analysis cycle, print report, then return."""
    run_cycle(_build_graph())


def start_scheduler() -> None:
    """Run immediately, then on two cadences (Mon–Fri ET):
    - Market hours  09:00–15:30 : every 30 minutes
    - After-hours   16:00–20:00 : every 60 minutes
    """
    graph = _build_graph()

    if _in_trading_window():
        run_cycle(graph)
    else:
        log.info("startup outside trading window — skipping initial cycle")

    scheduler = BlockingScheduler(timezone="America/New_York")

    # Market hours: every 30 min, 09:00–15:30 ET
    scheduler.add_job(
        run_cycle,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="*/30",
            timezone="America/New_York",
        ),
        args=[graph],
    )

    # After-hours: every 60 min, 16:00–20:00 ET
    scheduler.add_job(
        run_cycle,
        CronTrigger(
            day_of_week="mon-fri",
            hour="16-20",
            minute="0",
            timezone="America/New_York",
        ),
        args=[graph],
    )

    log.info(
        "scheduler running — market hours every 30 min (09:00–15:30 ET), "
        "after-hours every 60 min (16:00–20:00 ET)"
    )
    scheduler.start()
