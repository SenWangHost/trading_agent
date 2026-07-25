import os
from unittest.mock import MagicMock, patch

from state import AgentState, TechnicalSignal

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


def _make_bars_response(n: int = 60) -> dict:
    return {
        "status": "OK",
        "results": [
            {
                "c": 190.0 + i * 0.1,
                "o": 189.5 + i * 0.1,
                "h": 191.0 + i * 0.1,
                "l": 189.0 + i * 0.1,
                "v": 1_000_000,
                "t": 1700000000000 + i * 300_000,
            }
            for i in range(n)
        ],
    }


def _mock_get(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_data
    return resp


def test_run_technical_returns_signal():
    fake_signal = TechnicalSignal(
        ticker="AAPL", direction="bullish", confidence=0.75,
        rsi=48.0, macd_signal="bullish", current_price=196.5,
        reasoning="RSI neutral, MACD bullish crossover",
    )

    # First call: daily bars for indicators. Second call: latest minute bar (snapshot price).
    snapshot_response = {"status": "OK", "results": [{"c": 196.5, "t": 1700000000000}]}
    mock_get = MagicMock(side_effect=[
        _mock_get(_make_bars_response()),
        _mock_get(snapshot_response),
    ])

    with patch("polygon_client.requests.get", mock_get), \
         patch("agents.technical.build_llm") as MockLLM:

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_signal
        MockLLM.return_value = mock_llm

        from agents.technical import run_technical
        result = run_technical(_make_state())

    assert "technical_signals" in result
    assert "AAPL" in result["technical_signals"]
    assert result["technical_signals"]["AAPL"].direction == "bullish"
    assert "prices" in result
    assert result["prices"]["AAPL"] == 196.5  # snapshot price, not prev close


def test_run_technical_returns_neutral_on_api_failure():
    with patch("polygon_client.requests.get") as mock_get, \
         patch("agents.technical.build_llm"):
        mock_get.side_effect = Exception("API down")

        from agents.technical import run_technical
        result = run_technical(_make_state())

    signal = result["technical_signals"]["AAPL"]
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
    assert "API down" in signal.reasoning
