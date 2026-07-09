import pytest
from unittest.mock import MagicMock, patch
from broker.base import Position, OrderResult
from broker.alpaca import AlpacaBroker
from broker.robinhood import RobinhoodBroker
from broker.ibkr import IBKRBroker


@pytest.fixture
def mock_trading_client():
    with patch("broker.alpaca.TradingClient") as MockClient:
        client = MagicMock()
        MockClient.return_value = client

        pos = MagicMock()
        pos.symbol = "AAPL"
        pos.qty = "10"
        pos.avg_entry_price = "190.00"
        pos.current_price = "195.00"
        client.get_all_positions.return_value = [pos]

        account = MagicMock()
        account.portfolio_value = "100000.00"
        client.get_account.return_value = account

        order = MagicMock()
        order.id = "order-123"
        order.status = "accepted"
        client.submit_order.return_value = order

        yield client


def test_alpaca_get_positions(mock_trading_client):
    broker = AlpacaBroker("key", "secret", paper=True)
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].ticker == "AAPL"
    assert positions[0].qty == 10.0
    assert positions[0].current_price == 195.0


def test_alpaca_get_portfolio_value(mock_trading_client):
    broker = AlpacaBroker("key", "secret", paper=True)
    assert broker.get_portfolio_value() == 100000.0


def test_alpaca_place_order_buy(mock_trading_client):
    broker = AlpacaBroker("key", "secret", paper=True)
    result = broker.place_order("AAPL", "buy", 5.0)
    assert result.order_id == "order-123"
    assert result.action == "buy"
    assert result.status == "accepted"


def test_alpaca_place_order_sell(mock_trading_client):
    broker = AlpacaBroker("key", "secret", paper=True)
    result = broker.place_order("AAPL", "sell", 5.0)
    assert result.action == "sell"


def test_robinhood_stub_raises():
    broker = RobinhoodBroker()
    with pytest.raises(NotImplementedError):
        broker.get_portfolio_value()


def test_ibkr_stub_raises():
    broker = IBKRBroker()
    with pytest.raises(NotImplementedError):
        broker.place_order("AAPL", "buy", 1.0)
