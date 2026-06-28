from unittest.mock import MagicMock, patch
from state import AgentState, TechnicalSignal, FundamentalSignal, NewsSignal, TradeDecision
from broker.base import BaseBroker, Position, OrderResult


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
        pe_ratio=28.0, pb_ratio=None, market_cap=3e12,
        week_52_high=220.0, week_52_low=165.0, reasoning="ok",
    )
    news = NewsSignal(
        ticker=ticker, direction="bullish", confidence=0.7,
        material_event=False, reasoning="ok",
    )
    return tech, fund, news


def test_build_graph_returns_runnable():
    from graph import build_graph
    broker = StubBroker()
    g = build_graph(broker)
    assert hasattr(g, "invoke")


def test_graph_full_cycle_places_buy_order():
    tech, fund, news = _make_signals("AAPL")
    buy_decision = TradeDecision(ticker="AAPL", action="buy", size_pct=5.0, rationale="all bullish")

    with patch("agents.technical.rh") as mock_rh, \
         patch("agents.technical.ChatAnthropic") as MockTechLLM, \
         patch("agents.fundamental.rh") as mock_frh, \
         patch("agents.fundamental.ChatAnthropic") as MockFundLLM, \
         patch("agents.news.requests.get") as mock_get, \
         patch("agents.news.ChatAnthropic") as MockNewsLLM, \
         patch("agents.supervisor.ChatAnthropic") as MockSupLLM:

        mock_rh.stocks.get_stock_historicals.return_value = [
            {"close_price": str(190 + i), "open_price": "189", "high_price": "192",
             "low_price": "188", "volume": "1000000", "begins_at": "2026-06-28T10:00:00Z"}
            for i in range(60)
        ]
        _setup_llm(MockTechLLM, tech)

        mock_frh.stocks.get_fundamentals.return_value = [{"pe_ratio": "28.0"}]
        _setup_llm(MockFundLLM, fund)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"published_utc": "2026-06-28T09:00:00Z",
                                                     "title": "Good news", "description": "Positive"}]}
        mock_get.return_value = mock_resp
        _setup_llm(MockNewsLLM, news)

        _setup_llm(MockSupLLM, buy_decision)

        broker = StubBroker()
        from graph import build_graph
        g = build_graph(broker)
        initial_state: AgentState = {
            "tickers": ["AAPL"],
            "current_ticker": "",
            "prices": {},
            "technical_signals": {},
            "fundamental_signals": {},
            "news_signals": {},
            "decisions": [],
            "cycle_timestamp": "2026-06-28T10:00:00",
        }
        result = g.invoke(initial_state)

    assert len(result["decisions"]) == 1
    assert result["decisions"][0].action == "buy"
    assert len(broker.orders) == 1


def _setup_llm(MockLLM, return_value):
    instance = MagicMock()
    instance.with_structured_output.return_value.invoke.return_value = return_value
    MockLLM.return_value = instance
