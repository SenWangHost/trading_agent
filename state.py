import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel


class PortfolioPosition(BaseModel):
    ticker: str
    quantity: float
    average_buy_price: float
    current_value: float
    equity_change_pct: float


class TechnicalSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    rsi: float
    macd_signal: str
    current_price: float
    reasoning: str


class FundamentalSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    pe_ratio: float | None
    pb_ratio: float | None
    market_cap: float | None
    week_52_high: float | None
    week_52_low: float | None
    reasoning: str


class NewsSignal(BaseModel):
    ticker: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    material_event: bool
    reasoning: str


class RecentOrder(BaseModel):
    order_id: str
    ticker: str
    action: str
    order_type: str
    qty: float
    filled_qty: float
    limit_price: float | None
    status: str
    submitted_at: str
    filled_at: str | None


class TradeDecision(BaseModel):
    ticker: str
    action: Literal["buy", "sell", "hold"]
    size_pct: float
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None  # set when order_type == "limit"
    rationale: str


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class AgentState(TypedDict):
    tickers: list[str]
    current_ticker: str
    portfolio_positions: Annotated[dict[str, PortfolioPosition], _merge_dicts]
    recent_orders: list[RecentOrder]
    prices: Annotated[dict[str, float], _merge_dicts]
    technical_signals: Annotated[dict[str, TechnicalSignal], _merge_dicts]
    fundamental_signals: Annotated[dict[str, FundamentalSignal], _merge_dicts]
    news_signals: Annotated[dict[str, NewsSignal], _merge_dicts]
    decisions: Annotated[list[TradeDecision], operator.add]
    cycle_timestamp: str
    report: str
