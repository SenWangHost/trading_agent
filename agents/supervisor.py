import logging

from langchain_anthropic import ChatAnthropic

from state import AgentState, FundamentalSignal, NewsSignal, PortfolioPosition, TechnicalSignal, TradeDecision

log = logging.getLogger(__name__)


def run_supervisor(state: AgentState) -> dict:
    tickers = state["tickers"]
    log.info("supervisor → start (%d tickers)", len(tickers))
    decisions = []
    portfolio_positions = state.get("portfolio_positions", {})
    for ticker in tickers:
        tech = state["technical_signals"].get(ticker)
        fund = state["fundamental_signals"].get(ticker)
        news = state["news_signals"].get(ticker)
        pos = portfolio_positions.get(ticker)
        try:
            llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(TradeDecision)
            decision = llm.invoke(_build_prompt(ticker, tech, fund, news, pos))
            log.info("supervisor %s → %s %.1f%%", ticker, decision.action, decision.size_pct)
        except Exception as e:
            log.warning("supervisor %s → LLM failed, defaulting to hold: %s", ticker, e)
            decision = TradeDecision(
                ticker=ticker, action="hold", size_pct=0.0, rationale="LLM synthesis failed."
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
) -> str:
    if pos:
        position_context = (
            f"Current position: {pos.quantity:.4f} shares, avg cost ${pos.average_buy_price:.2f}, "
            f"current value ${pos.current_value:.2f}, unrealized P&L {pos.equity_change_pct:+.2f}%"
        )
    else:
        position_context = "No current position held."

    return f"""You are reviewing a real Robinhood portfolio position for {ticker}.
Recommend whether to add to the position (buy), reduce or exit it (sell), or make no change (hold).

{position_context}

Guidelines:
- action "buy": add to the existing position. size_pct = percentage of total portfolio to add.
- action "sell": trim or fully exit the position. size_pct = percentage of total portfolio to reduce.
- action "hold": no change. size_pct must be 0.
- Keep any single position under 15% of total portfolio.
- Default to hold when signals conflict or data is incomplete.
- When a position has significant unrealized gains and signals turn bearish, lean toward sell to protect profits.
- When a position has significant unrealized losses and signals remain bearish, lean toward sell to limit further downside.

Technical signal: {tech.model_dump_json() if tech else 'unavailable'}
Fundamental signal: {fund.model_dump_json() if fund else 'unavailable'}
News signal: {news.model_dump_json() if news else 'unavailable'}"""
