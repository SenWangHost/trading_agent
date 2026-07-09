import logging

from langchain_anthropic import ChatAnthropic

from state import AgentState, FundamentalSignal, NewsSignal, PortfolioPosition, RecentOrder, TechnicalSignal, TradeDecision

log = logging.getLogger(__name__)


def run_supervisor(state: AgentState) -> dict:
    tickers = state["tickers"]
    log.info("supervisor → start (%d tickers)", len(tickers))
    decisions = []
    portfolio_positions = state.get("portfolio_positions", {})
    recent_orders = state.get("recent_orders", [])
    for ticker in tickers:
        tech = state["technical_signals"].get(ticker)
        fund = state["fundamental_signals"].get(ticker)
        news = state["news_signals"].get(ticker)
        pos = portfolio_positions.get(ticker)
        ticker_orders = [o for o in recent_orders if o.ticker == ticker]
        try:
            llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(TradeDecision)
            decision = llm.invoke(_build_prompt(ticker, tech, fund, news, pos, ticker_orders))
            log.info("supervisor %s → %s %.1f%%", ticker, decision.action, decision.size_pct)
        except Exception as e:
            log.warning("supervisor %s → LLM failed, defaulting to hold: %s", ticker, e)
            decision = TradeDecision(
                ticker=ticker, action="hold", size_pct=0.0,
                rationale="LLM synthesis failed.",
            )
        decisions.append(decision)
    log.info("supervisor → done")
    return {"decisions": decisions}


def _build_prompt(
    ticker: str,
    tech: TechnicalSignal | None,
    fund: FundamentalSignal | None,
    news: NewsSignal | None,
    pos: PortfolioPosition | None,
    recent_orders: list[RecentOrder] | None = None,
) -> str:
    if pos:
        position_context = (
            f"Current position: {pos.quantity:.4f} shares, avg cost ${pos.average_buy_price:.2f}, "
            f"current value ${pos.current_value:.2f}, unrealized P&L {pos.equity_change_pct:+.2f}%"
        )
    else:
        position_context = "No current position held."

    current_price = tech.current_price if tech else None
    price_context = f"Current price: ${current_price:.2f}" if current_price else "Current price: unavailable"

    if recent_orders:
        order_lines = []
        for o in recent_orders[:10]:
            price_tag = f" @ ${o.limit_price:.2f}" if o.limit_price else ""
            filled_tag = f", filled {o.filled_qty:.0f}/{o.qty:.0f} shares" if o.filled_qty > 0 else ""
            order_lines.append(
                f"  - {o.submitted_at[:10]} {o.action.upper()} {o.qty:.0f}sh"
                f" [{o.order_type}{price_tag}] → {o.status}{filled_tag}"
            )
        orders_context = "Recent orders (last 7 days):\n" + "\n".join(order_lines)
    else:
        orders_context = "Recent orders: none in the last 7 days."

    return f"""You are a swing trading supervisor reviewing a US stock position for {ticker}.
Decide whether to buy, sell, or hold, and specify the exact order to place.

{position_context}
{price_context}
{orders_context}

## Action guidelines
- "buy"  — enter or add to position. size_pct = % of total portfolio to deploy.
- "sell" — trim or fully exit position. size_pct = % of total portfolio to reduce.
- "hold" — no change. size_pct must be 0.
- Keep any single position under 15% of total portfolio.
- Default to hold when signals conflict or data is incomplete.
- Lean sell when unrealized gains are significant and signals turn bearish (protect profits).
- Lean sell when unrealized losses are significant and signals stay bearish (limit downside).

## Order type guidelines (applies to buy and sell only)
Prefer limit orders for swing trading to control fill price and avoid slippage.

- order_type "limit": use for most buys and sells under normal conditions.
  - limit_price for BUY: set 0.2–0.5% below current price, or at the nearest technical support level.
  - limit_price for SELL: set 0.2–0.5% above current price, or at the nearest technical resistance level.
- order_type "market": use only when speed matters more than price.
  - material_event = true in news signal (earnings surprise, FDA decision, merger).
  - Strong breakout with high momentum — missing the move costs more than slippage.
  - Urgent stop-loss exit where the position is deteriorating rapidly.

Technical signal: {tech.model_dump_json() if tech else 'unavailable'}
Fundamental signal: {fund.model_dump_json() if fund else 'unavailable'}
News signal: {news.model_dump_json() if news else 'unavailable'}"""
