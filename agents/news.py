import os

import requests
from langchain_anthropic import ChatAnthropic

from state import AgentState, NewsSignal

_POLYGON_URL = "https://api.polygon.io/v2/reference/news"


def run_news(state: AgentState) -> dict:
    ticker = state["current_ticker"]
    try:
        articles = _fetch(ticker)
        summary = _format(ticker, articles)
        llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(NewsSignal)
        signal = llm.invoke(
            f"Analyze these recent news items for {ticker} and return a structured sentiment signal.\n\n{summary}"
        )
    except Exception as e:
        signal = NewsSignal(
            ticker=ticker,
            direction="neutral",
            confidence=0.0,
            material_event=False,
            reasoning=f"News fetch failed: {e}",
        )
    return {"news_signals": {ticker: signal}}


def _fetch(ticker: str, limit: int = 10) -> list[dict]:
    resp = requests.get(
        _POLYGON_URL,
        params={"ticker": ticker, "limit": limit, "apiKey": os.environ["POLYGON_API_KEY"]},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def _format(ticker: str, articles: list[dict]) -> str:
    if not articles:
        return f"No recent news found for {ticker}."
    lines = [f"Recent news for {ticker} ({len(articles)} articles):"]
    for a in articles:
        date = a.get("published_utc", "")[:10]
        title = a.get("title", "")
        desc = str(a.get("description", ""))[:150]
        lines.append(f"- [{date}] {title} | {desc}")
    return "\n".join(lines)
