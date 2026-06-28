from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from broker.base import BaseBroker, OrderResult, Position


class AlpacaBroker(BaseBroker):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_positions(self) -> list[Position]:
        return [
            Position(
                ticker=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
            )
            for p in self.client.get_all_positions()
        ]

    def place_order(self, ticker: str, action: str, qty: float) -> OrderResult:
        side = OrderSide.BUY if action == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(req)
        return OrderResult(
            order_id=str(order.id),
            ticker=ticker,
            action=action,
            qty=qty,
            status=str(order.status),
        )

    def get_portfolio_value(self) -> float:
        return float(self.client.get_account().portfolio_value)
