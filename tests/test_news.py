import os
from unittest.mock import MagicMock, patch

from state import AgentState, NewsSignal

os.environ.setdefault("POLYGON_API_KEY", "test_polygon_key")


def _make_state(ticker: str = "AAPL") -> AgentState:
    return AgentState(
        tickers=[ticker],
        current_ticker=ticker,
        portfolio_positions={},
        recent_orders=[],
        prices={},
        technical_signals={},
        fundamental_signals={},
        news_signals={},
        decisions=[],
        cycle_timestamp="2026-06-28T10:00:00",
        report="",
    )


def _make_news_response(articles: list | None = None) -> dict:
    if articles is None:
        articles = [
            {
                "title": "Apple beats earnings",
                "description": "Apple reported record quarterly revenue.",
                "published_utc": "2026-06-28T09:00:00Z",
            },
            {
                "title": "AAPL upgrade to buy",
                "description": "Analyst upgrades Apple to buy with $220 target.",
                "published_utc": "2026-06-27T14:00:00Z",
            },
        ]
    return {"status": "OK", "results": articles}


def _mock_get(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_data
    return resp


def test_run_news_returns_signal():
    fake_signal = NewsSignal(
        ticker="AAPL", direction="bullish", confidence=0.8,
        material_event=True, reasoning="Earnings beat is a material positive event",
    )

    with patch("polygon_client.requests.get", return_value=_mock_get(_make_news_response())), \
         patch("agents.news.ChatAnthropic") as MockLLM:

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = fake_signal
        MockLLM.return_value = mock_llm

        from agents.news import run_news
        result = run_news(_make_state())

    assert "news_signals" in result
    assert result["news_signals"]["AAPL"].direction == "bullish"
    assert result["news_signals"]["AAPL"].material_event is True


def test_run_news_returns_neutral_on_api_failure():
    with patch("polygon_client.requests.get") as mock_get, \
         patch("agents.news.ChatAnthropic"):
        mock_get.side_effect = Exception("connection refused")

        from agents.news import run_news
        result = run_news(_make_state())

    signal = result["news_signals"]["AAPL"]
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
    assert signal.material_event is False
