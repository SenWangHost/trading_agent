import datetime

import robin_stocks.robinhood as rh
from langchain_anthropic import ChatAnthropic

from state import AgentState, FundamentalSignal

_cache: dict[str, tuple[str, FundamentalSignal]] = {}


def run_fundamental(state: AgentState) -> dict:
    ticker = state["current_ticker"]
    today = datetime.date.today().isoformat()

    if ticker in _cache and _cache[ticker][0] == today:
        return {"fundamental_signals": {ticker: _cache[ticker][1]}}

    try:
        data = rh.stocks.get_fundamentals(ticker)[0]
        summary = _format(ticker, data)
        llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(FundamentalSignal)
        signal = llm.invoke(
            f"Analyze these fundamental metrics for {ticker} and return a structured signal.\n\n{summary}"
        )
    except Exception as e:
        signal = FundamentalSignal(
            ticker=ticker,
            direction="neutral",
            confidence=0.0,
            pe_ratio=None,
            pb_ratio=None,
            market_cap=None,
            week_52_high=None,
            week_52_low=None,
            reasoning=f"Data fetch failed: {e}",
        )

    _cache[ticker] = (today, signal)
    return {"fundamental_signals": {ticker: signal}}


def _fmt(data: dict, key: str) -> str:
    val = data.get(key)
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _format(ticker: str, data: dict) -> str:
    return (
        f"Ticker: {ticker}\n"
        f"PE ratio: {_fmt(data, 'pe_ratio')}\n"
        f"PB ratio: {_fmt(data, 'pb_ratio')}\n"
        f"Market cap: {_fmt(data, 'market_cap')}\n"
        f"52-week high: {_fmt(data, 'high_52_weeks')}\n"
        f"52-week low: {_fmt(data, 'low_52_weeks')}\n"
        f"Dividend yield: {_fmt(data, 'dividend_yield')}\n"
        f"Description: {str(data.get('description', ''))[:300]}"
    )
