import datetime
import os
from unittest.mock import MagicMock, patch

from state import AgentState, PortfolioPosition, TechnicalSignal, FundamentalSignal, NewsSignal, TradeDecision
from broker.base import BaseBroker, Position, OrderResult

os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_secret")
os.environ.setdefault("POLYGON_API_KEY", "test_polygon_key")


class StubBroker(BaseBroker):
    def __init__(self):
        self.orders = []

    def get_positions(self) -> list[Position]:
        return []

    def place_order(self, ticker: str, action: str, qty: float) -> OrderResult:
        self.orders.append((ticker, action, qty))
        return OrderResult(order_id="stub-1", ticker=ticker, action=action, qty=qty, status="filled")

    def get_portfolio_value(self) -> float:
        return 100_000.0


def _make_signals(ticker: str):
    tech = TechnicalSignal(
        ticker=ticker, direction="bullish", confidence=0.8,
        rsi=45.0, macd_signal="bullish", current_price=195.0, reasoning="ok",
    )
    fund = FundamentalSignal(
        ticker=ticker, direction="bullish", confidence=0.6,
        pe_ratio=None, pb_ratio=None, market_cap=3e12,
        week_52_high=None, week_52_low=None, reasoning="ok",
    )
    news = NewsSignal(
        ticker=ticker, direction="bullish", confidence=0.7,
        material_event=False, reasoning="ok",
    )
    return tech, fund, news


def _dispatch_polygon_get(url, **kwargs):
    """Single requests.get mock that routes all three Polygon data endpoints."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()

    if "/v2/aggs/" in url:  # bars (technical)
        resp.json.return_value = {
            "status": "OK",
            "results": [
                {"c": 190.0 + i * 0.1, "o": 189.5 + i * 0.1,
                 "h": 191.0 + i * 0.1, "l": 189.0 + i * 0.1,
                 "v": 1_000_000, "t": 1700000000000 + i * 300_000}
                for i in range(60)
            ],
        }
    elif "/v3/reference/tickers" in url:  # ticker details (fundamental)
        resp.json.return_value = {
            "results": {
                "market_cap": 3e12,
                "description": "Apple Inc.",
                "share_class_shares_outstanding": 15e9,
            }
        }
    elif "/vX/reference/financials" in url:  # financials (fundamental)
        resp.json.return_value = {
            "results": [{
                "financials": {
                    "income_statement": {
                        "revenues": {"value": 100e9},
                        "net_income_loss": {"value": 25e9},
                        "basic_earnings_per_share": {"value": 1.64},
                    },
                    "balance_sheet": {
                        "equity": {"value": 62e9},
                        "assets": {"value": 350e9},
                    },
                }
            }]
        }
    else:  # /v2/reference/news (news)
        resp.json.return_value = {
            "status": "OK",
            "results": [
                {"title": "Good news", "description": "Positive outlook.",
                 "published_utc": "2026-06-28T09:00:00Z"},
            ],
        }
    return resp


def test_build_graph_returns_runnable():
    from graph import build_graph
    broker = StubBroker()
    g = build_graph(broker)
    assert hasattr(g, "invoke")


def test_graph_full_cycle_generates_report():
    tech, fund, news = _make_signals("AAPL")
    buy_decision = TradeDecision(ticker="AAPL", action="buy", size_pct=5.0, rationale="all bullish")

    mock_yf_ticker = MagicMock()
    mock_yf_ticker.calendar = {
        "Earnings Date": [datetime.date(2026, 7, 30)],
        "Earnings Average": 1.89,
        "Earnings Low": 1.83,
        "Earnings High": 1.99,
        "Revenue Average": 108_900_000_000,
    }

    with patch("requests.get", side_effect=_dispatch_polygon_get), \
         patch("agents.fundamental.yf") as mock_yf, \
         patch("agents.technical.ChatAnthropic") as MockTechLLM, \
         patch("agents.fundamental.ChatAnthropic") as MockFundLLM, \
         patch("agents.news.ChatAnthropic") as MockNewsLLM, \
         patch("agents.supervisor.ChatAnthropic") as MockSupLLM:

        mock_yf.Ticker.return_value = mock_yf_ticker

        _setup_llm(MockTechLLM, tech)
        _setup_llm(MockFundLLM, fund)
        _setup_llm(MockNewsLLM, news)
        _setup_llm(MockSupLLM, buy_decision)

        broker = StubBroker()
        from graph import build_graph
        g = build_graph(broker)
        initial_state: AgentState = {
            "tickers": ["AAPL"],
            "current_ticker": "",
            "portfolio_positions": {
                "AAPL": PortfolioPosition(
                    ticker="AAPL", quantity=5.0, average_buy_price=167.45,
                    current_value=911.55, equity_change_pct=8.87,
                )
            },
            "prices": {},
            "technical_signals": {},
            "fundamental_signals": {},
            "news_signals": {},
            "decisions": [],
            "cycle_timestamp": "2026-06-28T10:00:00",
            "report": "",
        }
        result = g.invoke(initial_state)

    assert len(result["decisions"]) == 1
    assert result["decisions"][0].action == "buy"
    assert len(broker.orders) == 0  # existing AAPL position skips buy (swing no-double-up guard)
    assert "AAPL" in result["report"]
    assert "BUY" in result["report"]
    assert "Trading Decision Report" in result["report"]


def _setup_llm(MockLLM, return_value):
    instance = MagicMock()
    instance.with_structured_output.return_value.invoke.return_value = return_value
    MockLLM.return_value = instance
