import datetime
import json
import os
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from broker.alpaca import AlpacaBroker
from graph import build_graph
from state import AgentState


def run_cycle(tickers: list[str], graph) -> None:
    now = datetime.datetime.utcnow()
    print(f"\n[{now.isoformat()}] Starting cycle for: {', '.join(tickers)}")

    initial_state: AgentState = {
        "tickers": tickers,
        "current_ticker": "",
        "prices": {},
        "technical_signals": {},
        "fundamental_signals": {},
        "news_signals": {},
        "decisions": [],
        "cycle_timestamp": now.isoformat(),
    }

    result = graph.invoke(initial_state)
    _log(result)
    _print_summary(result)


def _log(state: AgentState) -> None:
    log_path = Path("logs/decisions.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": state["cycle_timestamp"],
        "tickers": state["tickers"],
        "technical_signals": {k: v.model_dump() for k, v in state["technical_signals"].items()},
        "fundamental_signals": {k: v.model_dump() for k, v in state["fundamental_signals"].items()},
        "news_signals": {k: v.model_dump() for k, v in state["news_signals"].items()},
        "decisions": [d.model_dump() for d in state["decisions"]],
    }
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _print_summary(state: AgentState) -> None:
    print(f"{'─' * 60}")
    for d in state["decisions"]:
        tech = state["technical_signals"].get(d.ticker)
        fund = state["fundamental_signals"].get(d.ticker)
        news = state["news_signals"].get(d.ticker)
        print(
            f"  {d.ticker:6s} → {d.action.upper():4s} {d.size_pct:.1f}%"
            f" | tech={tech.direction if tech else '?'}"
            f" | fund={fund.direction if fund else '?'}"
            f" | news={news.direction if news else '?'}"
        )
        print(f"         rationale: {d.rationale[:100]}")
    print(f"{'─' * 60}")


def start_scheduler(tickers: list[str]) -> None:
    broker = AlpacaBroker(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )
    graph = build_graph(broker)

    run_cycle(tickers, graph)

    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(
        run_cycle,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/30",
                    timezone="America/New_York"),
        args=[tickers, graph],
    )
    print(f"Scheduler running. Next cycle at market hours (Mon–Fri 9:00–15:30 ET).")
    scheduler.start()
