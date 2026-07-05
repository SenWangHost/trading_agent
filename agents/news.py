import logging
import os
from datetime import datetime, timedelta

from langchain_anthropic import ChatAnthropic

import polygon_client

from state import AgentState, NewsSignal

_POLYGON_BASE = "https://api.polygon.io"
log = logging.getLogger(__name__)


def run_news(state: AgentState) -> dict:
    ticker = state["current_ticker"]
    log.info("%s news → start", ticker)
    try:
        articles = _fetch(ticker)
        log.info("%s news → %d articles fetched", ticker, len(articles))
        summary = _format(ticker, articles)
        llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(NewsSignal)
        signal = llm.invoke(
            f"Analyze these recent news items for {ticker} and return a structured sentiment signal.\n\n{summary}"
        )
        log.info("%s news → done | %s confidence=%.2f material_event=%s",
                 ticker, signal.direction, signal.confidence, signal.material_event)
    except Exception as e:
        log.warning("%s news → failed: %s", ticker, e)
        signal = NewsSignal(
            ticker=ticker,
            direction="neutral",
            confidence=0.0,
            material_event=False,
            reasoning=f"News fetch failed: {e}",
        )
    return {"news_signals": {ticker: signal}}


def _fetch(ticker: str, limit: int = 10) -> list:
    api_key = os.environ["POLYGON_API_KEY"]
    since = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    resp = polygon_client.get(
        f"{_POLYGON_BASE}/v2/reference/news",
        params={
            "ticker": ticker,
            "limit": limit,
            "order": "desc",
            "sort": "published_utc",
            "published_utc.gte": since,
            "apiKey": api_key,
        },
        timeout=10,
    )
    return resp.json().get("results", [])


def _format(ticker: str, articles: list) -> str:
    if not articles:
        return f"No recent news found for {ticker}."
    lines = [f"Recent news for {ticker} ({len(articles)} articles):"]
    for a in articles:
        date = str(a.get("published_utc", ""))[:10]
        title = a.get("title", "")
        description = str(a.get("description", ""))[:150]
        lines.append(f"- [{date}] {title} | {description}")
    return "\n".join(lines)
