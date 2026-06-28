from unittest.mock import MagicMock, patch
from state import AgentState, TechnicalSignal, FundamentalSignal, NewsSignal, TradeDecision


def _make_state_with_signals() -> AgentState:
    tech = TechnicalSignal(
        ticker="AAPL", direction="bullish", confidence=0.8,
        rsi=45.0, macd_signal="bullish", current_price=195.0,
        reasoning="RSI neutral, MACD bullish",
    )
    fund = FundamentalSignal(
        ticker="AAPL", direction="bullish", confidence=0.6,
        pe_ratio=28.5, pb_ratio=None, market_cap=3e12,
        week_52_high=220.0, week_52_low=165.0, reasoning="Reasonable PE",
    )
    news = NewsSignal(
        ticker="AAPL", direction="bullish", confidence=0.75,
        material_event=False, reasoning="Positive analyst coverage",
    )
    return AgentState(
        tickers=["AAPL"],
        current_ticker="",
        prices={"AAPL": 195.0},
        technical_signals={"AAPL": tech},
        fundamental_signals={"AAPL": fund},
        news_signals={"AAPL": news},
        decisions=[],
        cycle_timestamp="2026-06-28T10:00:00",
    )


def test_run_supervisor_returns_decision():
    fake_decision = TradeDecision(
        ticker="AAPL", action="buy", size_pct=5.0, rationale="All three signals bullish",
    )

    with patch("agents.supervisor.ChatAnthropic") as MockLLM:
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.return_value = fake_decision
        MockLLM.return_value = mock_llm_instance

        from agents.supervisor import run_supervisor
        result = run_supervisor(_make_state_with_signals())

    assert "decisions" in result
    assert len(result["decisions"]) == 1
    assert result["decisions"][0].action == "buy"
    assert result["decisions"][0].ticker == "AAPL"


def test_run_supervisor_holds_on_llm_failure():
    with patch("agents.supervisor.ChatAnthropic") as MockLLM:
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.side_effect = Exception("LLM error")
        MockLLM.return_value = mock_llm_instance

        from agents.supervisor import run_supervisor
        result = run_supervisor(_make_state_with_signals())

    assert result["decisions"][0].action == "hold"
    assert result["decisions"][0].size_pct == 0.0


def test_run_supervisor_handles_missing_signals():
    state = AgentState(
        tickers=["AAPL"],
        current_ticker="",
        prices={},
        technical_signals={},
        fundamental_signals={},
        news_signals={},
        decisions=[],
        cycle_timestamp="2026-06-28T10:00:00",
    )
    fake_decision = TradeDecision(
        ticker="AAPL", action="hold", size_pct=0.0, rationale="Insufficient data",
    )

    with patch("agents.supervisor.ChatAnthropic") as MockLLM:
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value.invoke.return_value = fake_decision
        MockLLM.return_value = mock_llm_instance

        from agents.supervisor import run_supervisor
        result = run_supervisor(state)

    assert result["decisions"][0].action == "hold"
