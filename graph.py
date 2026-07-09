import logging
import math

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.fundamental import run_fundamental
from agents.news import run_news
from agents.supervisor import run_supervisor
from agents.technical import run_technical
from broker.base import BaseBroker
from state import AgentState

log = logging.getLogger(__name__)


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

    def generate_report(state: AgentState) -> dict:
        log.info("report_generator → start (%d tickers)", len(state["tickers"]))
        positions = state.get("portfolio_positions", {})
        total_value = sum(p.current_value for p in positions.values())

        lines = [
            "# Trading Decision Report",
            f"**Generated:** {state['cycle_timestamp']}",
            f"**Positions analyzed:** {', '.join(state['tickers'])}",
            "",
            "---",
            "",
            "## Portfolio Summary",
            f"**Total position value:** ${total_value:,.2f}",
            "",
        ]

        for decision in state["decisions"]:
            ticker = decision.ticker
            pos = positions.get(ticker)
            tech = state["technical_signals"].get(ticker)
            fund = state["fundamental_signals"].get(ticker)
            news = state["news_signals"].get(ticker)
            price = state["prices"].get(ticker, 0.0)
            ticker_orders = [o for o in state.get("recent_orders", []) if o.ticker == ticker]

            lines.append(f"## {ticker}")
            lines.append("")
            lines.append("**Current Position**")
            if pos:
                lines.append(f"- Shares held: {pos.quantity:.4f}")
                lines.append(f"- Avg buy price: ${pos.average_buy_price:.2f}")
                lines.append(f"- Current price: ${price:.2f}")
                lines.append(f"- Position value: ${pos.current_value:.2f}")
                lines.append(f"- P&L: {pos.equity_change_pct:+.2f}%")
            else:
                lines.append(f"- Current price: ${price:.2f}")
                lines.append("- No position held")
            lines.append("")
            lines.append("**Signals**")
            if tech:
                lines.append(f"- Technical: {tech.direction} (confidence {tech.confidence:.0%})")
                lines.append(f"  - RSI: {tech.rsi:.1f} | MACD: {tech.macd_signal}")
                lines.append(f"  - {tech.reasoning}")
            else:
                lines.append("- Technical: unavailable")
            if fund:
                lines.append(f"- Fundamental: {fund.direction} (confidence {fund.confidence:.0%})")
                pe = f"P/E {fund.pe_ratio:.1f}" if fund.pe_ratio else "P/E N/A"
                lines.append(f"  - {pe} | {fund.reasoning}")
            else:
                lines.append("- Fundamental: unavailable")
            if news:
                material = " ⚠ MATERIAL EVENT" if news.material_event else ""
                lines.append(f"- News: {news.direction} (confidence {news.confidence:.0%}){material}")
                lines.append(f"  - {news.reasoning}")
            else:
                lines.append("- News: unavailable")
            lines.append("")
            lines.append("**Recent Orders (7 days)**")
            if ticker_orders:
                for o in ticker_orders[:5]:
                    price_tag = f" @ ${o.limit_price:.2f}" if o.limit_price else ""
                    filled_tag = f", filled {o.filled_qty:.0f}/{o.qty:.0f}" if o.filled_qty > 0 else ""
                    lines.append(f"- {o.submitted_at[:10]} {o.action.upper()} {o.qty:.0f}sh [{o.order_type}{price_tag}] → {o.status}{filled_tag}")
            else:
                lines.append("- None")
            lines.append("")
            lines.append("**Recommendation**")
            lines.append(f"- Action: **{decision.action.upper()}**")
            lines.append(f"- Portfolio allocation: {decision.size_pct:.1f}%")
            order_detail = decision.order_type.upper()
            if decision.order_type == "limit" and decision.limit_price:
                order_detail += f" @ ${decision.limit_price:.2f}"
            lines.append(f"- Order: {order_detail}")
            lines.append(f"- Rationale: {decision.rationale}")
            lines.append("")
            lines.append("---")
            lines.append("")

        report = "\n".join(lines)
        log.info("report_generator → done (%d chars)", len(report))
        return {"report": report}

    def execute_trades(state: AgentState) -> dict:
        actionable = [d for d in state["decisions"] if d.action != "hold" and d.size_pct > 0]
        log.info("execute_trades → start (%d actionable decisions)", len(actionable))
        try:
            portfolio_value = broker.get_portfolio_value()
        except Exception as e:
            log.warning("execute_trades → could not fetch portfolio value: %s", e)
            return {}
        positions = state.get("portfolio_positions", {})
        for decision in state["decisions"]:
            if decision.action == "hold" or decision.size_pct <= 0:
                continue
            held = positions.get(decision.ticker)
            if decision.action == "buy" and held:
                log.info("%s buy → skipped (already held)", decision.ticker)
                continue
            if decision.action == "sell":
                if not held:
                    log.info("%s sell → skipped (no position held)", decision.ticker)
                    continue
                qty = held.quantity
            else:
                price = state["prices"].get(decision.ticker, 0.0)
                if price <= 0:
                    log.warning("%s %s → skipped (price unavailable)", decision.ticker, decision.action)
                    continue
                qty = math.floor((decision.size_pct / 100) * portfolio_value / price)
                if qty < 1:
                    log.info("%s %s → skipped (qty < 1 at current price)", decision.ticker, decision.action)
                    continue
            try:
                broker.place_order(
                    decision.ticker, decision.action, float(qty),
                    decision.order_type, decision.limit_price,
                )
                price_tag = f" @ ${decision.limit_price:.2f}" if decision.limit_price else ""
                log.info("%s %s %g shares [%s%s] → order submitted",
                         decision.ticker, decision.action, qty, decision.order_type, price_tag)
            except Exception as e:
                log.warning("%s %s → order failed: %s", decision.ticker, decision.action, e)
        log.info("execute_trades → done")
        return {}

    builder.add_node("dispatch", lambda state: state)
    builder.add_node("analyze_ticker", analyze_ticker)
    builder.add_node("supervisor", run_supervisor)
    builder.add_node("execute_trades", execute_trades)
    builder.add_node("report_generator", generate_report)

    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", dispatch_tickers)
    builder.add_edge("analyze_ticker", "supervisor")
    builder.add_edge("supervisor", "execute_trades")
    builder.add_edge("execute_trades", "report_generator")
    builder.add_edge("report_generator", END)

    return builder.compile()
