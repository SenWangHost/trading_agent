from unittest.mock import MagicMock, patch
from state import AgentState, TechnicalSignal


def _make_state(ticker: str = "AAPL") -> AgentState:
    return AgentState(
        tickers=[ticker],
        current_ticker=ticker,
        prices={},
        technical_signals={},
        fundamental_signals={},
        news_signals={},
        decisions=[],
        cycle_timestamp="2026-06-28T10:00:00",
    )


FAKE_HISTORICALS = [
    {"close_price": str(190 + i), "open_price": "189", "high_price": "192",
     "low_price": "188", "volume": "1000000", "begins_at": f"2026-06-28T{9+i//12:02d}:{(i*5)%60:02d}:00Z"}
    for i in range(60)
]


def test_run_technical_returns_signal():
    fake_signal = TechnicalSignal(
        ticker="AAPL",
        direction="bullish",
        confidence=0.75,
        rsi=48.0,
        macd_signal="bullish",
        current_price=195.0,
        reasoning="RSI neutral, MACD bullish crossover",
    )

    with patch("agents.technical.rh") as mock_rh, \
         patch("agents.technical.ChatAnthropic") as MockLLM:
        mock_rh.stocks.get_stock_historicals.return_value = FAKE_HISTORICALS

        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.return_value = fake_signal
        MockLLM.return_value = mock_llm_instance

        from agents.technical import run_technical
        result = run_technical(_make_state())

    assert "technical_signals" in result
    assert "AAPL" in result["technical_signals"]
    assert result["technical_signals"]["AAPL"].direction == "bullish"
    assert "prices" in result
    assert result["prices"]["AAPL"] > 0


def test_run_technical_returns_neutral_on_api_failure():
    with patch("agents.technical.rh") as mock_rh, \
         patch("agents.technical.ChatAnthropic"):
        mock_rh.stocks.get_stock_historicals.side_effect = Exception("API down")

        from agents.technical import run_technical
        result = run_technical(_make_state())

    signal = result["technical_signals"]["AAPL"]
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
    assert "API down" in signal.reasoning
