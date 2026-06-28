import math

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.fundamental import run_fundamental
from agents.news import run_news
from agents.supervisor import run_supervisor
from agents.technical import run_technical
from broker.base import BaseBroker
from state import AgentState


def build_graph(broker: BaseBroker):
    builder = StateGraph(AgentState)

    def dispatch_tickers(state: AgentState) -> list[Send]:
        return [
            Send("analyze_ticker", {**state, "current_ticker": ticker})
            for ticker in state["tickers"]
        ]

    def analyze_ticker(state: AgentState) -> dict:
        updates: dict = {}
        updates.update(run_technical(state))
        updates.update(run_fundamental(state))
        updates.update(run_news(state))
        return updates

    def execute_trades(state: AgentState) -> dict:
        try:
            portfolio_value = broker.get_portfolio_value()
        except Exception:
            return {}
        for decision in state["decisions"]:
            if decision.action == "hold" or decision.size_pct <= 0:
                continue
            price = state["prices"].get(decision.ticker, 0.0)
            if price <= 0:
                continue
            qty = math.floor((decision.size_pct / 100) * portfolio_value / price)
            if qty < 1:
                continue
            try:
                broker.place_order(decision.ticker, decision.action, float(qty))
            except Exception:
                pass
        return {}

    builder.add_node("dispatch", lambda state: state)
    builder.add_node("analyze_ticker", analyze_ticker)
    builder.add_node("supervisor", run_supervisor)
    builder.add_node("trade_executor", execute_trades)

    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", dispatch_tickers)
    builder.add_edge("analyze_ticker", "supervisor")
    builder.add_edge("supervisor", "trade_executor")
    builder.add_edge("trade_executor", END)

    return builder.compile()
