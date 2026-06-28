import datetime
from unittest.mock import MagicMock, patch
from state import AgentState, FundamentalSignal


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


FAKE_FUNDAMENTALS = [{
    "pe_ratio": "28.5",
    "pb_ratio": None,
    "market_cap": "3000000000000",
    "high_52_weeks": "220.0",
    "low_52_weeks": "165.0",
    "dividend_yield": "0.005",
    "description": "Apple Inc. designs consumer electronics.",
}]


def test_run_fundamental_returns_signal():
    fake_signal = FundamentalSignal(
        ticker="AAPL",
        direction="bullish",
        confidence=0.65,
        pe_ratio=28.5,
        pb_ratio=None,
        market_cap=3e12,
        week_52_high=220.0,
        week_52_low=165.0,
        reasoning="Reasonable PE, near 52-week high",
    )

    with patch("agents.fundamental.rh") as mock_rh, \
         patch("agents.fundamental.ChatAnthropic") as MockLLM:
        mock_rh.stocks.get_fundamentals.return_value = FAKE_FUNDAMENTALS

        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.return_value = fake_signal
        MockLLM.return_value = mock_llm_instance

        from agents.fundamental import run_fundamental
        result = run_fundamental(_make_state())

    assert "fundamental_signals" in result
    assert result["fundamental_signals"]["AAPL"].direction == "bullish"


def test_run_fundamental_uses_cache_on_second_call():
    fake_signal = FundamentalSignal(
        ticker="AAPL", direction="neutral", confidence=0.5,
        pe_ratio=None, pb_ratio=None, market_cap=None,
        week_52_high=None, week_52_low=None, reasoning="cached",
    )

    import agents.fundamental as mod
    today = datetime.date.today().isoformat()
    mod._cache["AAPL"] = (today, fake_signal)

    with patch("agents.fundamental.rh") as mock_rh, \
         patch("agents.fundamental.ChatAnthropic"):
        result = mod.run_fundamental(_make_state())
        mock_rh.stocks.get_fundamentals.assert_not_called()

    assert result["fundamental_signals"]["AAPL"].reasoning == "cached"

    del mod._cache["AAPL"]


def test_run_fundamental_returns_neutral_on_failure():
    with patch("agents.fundamental.rh") as mock_rh, \
         patch("agents.fundamental.ChatAnthropic"):
        mock_rh.stocks.get_fundamentals.side_effect = Exception("timeout")

        from agents.fundamental import run_fundamental
        result = run_fundamental(_make_state())

    signal = result["fundamental_signals"]["AAPL"]
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
