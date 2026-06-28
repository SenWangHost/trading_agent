from abc import ABC, abstractmethod
from pydantic import BaseModel


class Position(BaseModel):
    ticker: str
    qty: float
    avg_entry_price: float
    current_price: float


class OrderResult(BaseModel):
    order_id: str
    ticker: str
    action: str
    qty: float
    status: str


class BaseBroker(ABC):
    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def place_order(self, ticker: str, action: str, qty: float) -> OrderResult: ...

    @abstractmethod
    def get_portfolio_value(self) -> float: ...
