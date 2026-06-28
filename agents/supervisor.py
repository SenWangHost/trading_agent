from langchain_anthropic import ChatAnthropic

from state import AgentState, FundamentalSignal, NewsSignal, TechnicalSignal, TradeDecision


def run_supervisor(state: AgentState) -> dict:
    decisions = []
    for ticker in state["tickers"]:
        tech = state["technical_signals"].get(ticker)
        fund = state["fundamental_signals"].get(ticker)
        news = state["news_signals"].get(ticker)
        try:
            llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(TradeDecision)
            decision = llm.invoke(_build_prompt(ticker, tech, fund, news))
        except Exception:
            decision = TradeDecision(
                ticker=ticker, action="hold", size_pct=0.0, rationale="LLM synthesis failed."
            )
        decisions.append(decision)
    return {"decisions": decisions}


def _build_prompt(
    ticker: str,
    tech: TechnicalSignal | None,
    fund: FundamentalSignal | None,
    news: NewsSignal | None,
) -> str:
    return f"""You are a trading supervisor for a paper trading account.
Given the signals below for {ticker}, decide: buy, sell, or hold.
- size_pct is the percentage of total portfolio to allocate (0–10%).
- Default to hold with size_pct=0 when signals conflict or data is missing.
- Never exceed 10% per position.

Technical signal: {tech.model_dump_json() if tech else 'unavailable'}
Fundamental signal: {fund.model_dump_json() if fund else 'unavailable'}
News signal: {news.model_dump_json() if news else 'unavailable'}"""
