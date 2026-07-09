from broker.base import BaseBroker, OrderResult, Position


class IBKRBroker(BaseBroker):
    """Live IBKR broker. NOT ACTIVE — raises NotImplementedError on all methods.
    Implement and activate only after paper trading phase is validated."""

    def get_positions(self) -> list[Position]:
        raise NotImplementedError("IBKRBroker is not yet activated.")

    def place_order(self, ticker: str, action: str, qty: float) -> OrderResult:
        raise NotImplementedError("IBKRBroker is not yet activated.")

    def get_portfolio_value(self) -> float:
        raise NotImplementedError("IBKRBroker is not yet activated.")
