import datetime
import os
from unittest.mock import MagicMock, patch

import agents.fundamental as _fund_mod
from state import AgentState, FundamentalSignal

os.environ.setdefault("POLYGON_API_KEY", "test_polygon_key")


def setup_function():
    _fund_mod._cache.clear()


def _mock_yf_calendar():
    mock_ticker = MagicMock()
    mock_ticker.calendar = {
        "Earnings Date": [datetime.date(2026, 7, 30)],
        "Earnings Average": 1.89,
        "Earnings Low": 1.83,
        "Earnings High": 1.99,
        "Revenue Average": 108_900_000_000,
    }
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker
    return mock_yf


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


def _mock_polygon_get(url, **kwargs):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    if "reference/tickers" in url and "financials" not in url:
        resp.json.return_value = {
            "results": {
                "market_cap": 3_000_000_000_000.0,
                "description": "Apple Inc. designs consumer electronics.",
                "share_class_shares_outstanding": 15_000_000_000,
            }
        }
    else:
        resp.json.return_value = {
            "results": [{
                "financials": {
                    "income_statement": {
                        "revenues": {"value": 100_000_000_000.0},
                        "net_income_loss": {"value": 25_000_000_000.0},
                        "basic_earnings_per_share": {"value": 1.64},
                    },
                    "balance_sheet": {
                        "equity": {"value": 62_000_000_000.0},
                        "assets": {"value": 350_000_000_000.0},
                    },
                }
            }]
        }
    return resp


def test_run_fundamental_returns_signal():
    fake_signal = FundamentalSignal(
        ticker="AAPL", direction="bullish", confidence=0.65,
        pe_ratio=None, pb_ratio=None, market_cap=3e12,
        week_52_high=None, week_52_low=None,
        reasoning="Strong revenue, healthy margins",
    )

    with patch("polygon_client.requests.get", side_effect=_mock_polygon_get), \
         patch("agents.fundamental.yf", _mock_yf_calendar()), \
         patch("agents.fundamental.build_llm") as MockLLM:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_signal
        MockLLM.return_value = mock_llm

        from agents.fundamental import run_fundamental
        result = run_fundamental(_make_state())

    assert "fundamental_signals" in result
    assert result["fundamental_signals"]["AAPL"].direction == "bullish"


def test_run_fundamental_includes_earnings_in_prompt():
    fake_signal = FundamentalSignal(
        ticker="AAPL", direction="neutral", confidence=0.3,
        pe_ratio=None, pb_ratio=None, market_cap=3e12,
        week_52_high=None, week_52_low=None,
        reasoning="Earnings in 26 days — noted",
    )

    with patch("polygon_client.requests.get", side_effect=_mock_polygon_get), \
         patch("agents.fundamental.yf", _mock_yf_calendar()), \
         patch("agents.fundamental.build_llm") as MockLLM:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_signal
        MockLLM.return_value = mock_llm

        from agents.fundamental import run_fundamental
        run_fundamental(_make_state())

    prompt_text = mock_llm.invoke.call_args[0][0]
    assert "Earnings Calendar" in prompt_text
    assert "2026-07-30" in prompt_text


def test_run_fundamental_uses_cache_on_second_call():
    fake_signal = FundamentalSignal(
        ticker="AAPL", direction="neutral", confidence=0.5,
        pe_ratio=None, pb_ratio=None, market_cap=None,
        week_52_high=None, week_52_low=None, reasoning="cached",
    )

    import agents.fundamental as mod
    today = datetime.date.today().isoformat()
    mod._cache["AAPL"] = (today, fake_signal)

    with patch("polygon_client.requests.get") as mock_get, \
         patch("agents.fundamental.yf") as mock_yf, \
         patch("agents.fundamental.build_llm"):
        result = mod.run_fundamental(_make_state())
        mock_get.assert_not_called()
        mock_yf.Ticker.assert_not_called()

    assert result["fundamental_signals"]["AAPL"].reasoning == "cached"

    del mod._cache["AAPL"]


def test_run_fundamental_returns_neutral_on_failure():
    with patch("polygon_client.requests.get") as mock_get, \
         patch("agents.fundamental.yf"), \
         patch("agents.fundamental.build_llm"):
        mock_get.side_effect = Exception("timeout")

        from agents.fundamental import run_fundamental
        result = run_fundamental(_make_state())

    signal = result["fundamental_signals"]["AAPL"]
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
    assert "timeout" in signal.reasoning
