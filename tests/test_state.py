import pytest
from pydantic import ValidationError
from state import (
    TechnicalSignal,
    FundamentalSignal,
    NewsSignal,
    TradeDecision,
    AgentState,
    _merge_dicts,
)


def test_technical_signal_valid():
    s = TechnicalSignal(
        ticker="AAPL",
        direction="bullish",
        confidence=0.8,
        rsi=45.0,
        macd_signal="bullish",
        current_price=195.0,
        reasoning="RSI not overbought, MACD crossed up",
    )
    assert s.ticker == "AAPL"
    assert s.direction == "bullish"


def test_technical_signal_rejects_bad_direction():
    with pytest.raises(ValidationError):
        TechnicalSignal(
            ticker="AAPL",
            direction="up",  # invalid
            confidence=0.8,
            rsi=45.0,
            macd_signal="bullish",
            current_price=195.0,
            reasoning="x",
        )


def test_fundamental_signal_nullable_fields():
    s = FundamentalSignal(
        ticker="AAPL",
        direction="neutral",
        confidence=0.5,
        pe_ratio=None,
        pb_ratio=None,
        market_cap=None,
        week_52_high=None,
        week_52_low=None,
        reasoning="no data",
    )
    assert s.pe_ratio is None


def test_news_signal_valid():
    s = NewsSignal(
        ticker="AAPL",
        direction="bearish",
        confidence=0.7,
        material_event=True,
        reasoning="earnings miss",
    )
    assert s.material_event is True


def test_trade_decision_valid():
    d = TradeDecision(ticker="AAPL", action="buy", size_pct=5.0, rationale="strong signals")
    assert d.action == "buy"


def test_trade_decision_rejects_bad_action():
    with pytest.raises(ValidationError):
        TradeDecision(ticker="AAPL", action="short", size_pct=5.0, rationale="x")


def test_merge_dicts():
    a = {"AAPL": "sig_a"}
    b = {"MSFT": "sig_b"}
    assert _merge_dicts(a, b) == {"AAPL": "sig_a", "MSFT": "sig_b"}


def test_merge_dicts_overwrites_on_collision():
    a = {"AAPL": "old"}
    b = {"AAPL": "new"}
    assert _merge_dicts(a, b) == {"AAPL": "new"}
