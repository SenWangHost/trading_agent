import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel


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


class TradeDecision(BaseModel):
    ticker: str
    action: Literal["buy", "sell", "hold"]
    size_pct: float
    rationale: str


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class AgentState(TypedDict):
    tickers: list[str]
    current_ticker: str
    prices: Annotated[dict[str, float], _merge_dicts]
    technical_signals: Annotated[dict[str, TechnicalSignal], _merge_dicts]
    fundamental_signals: Annotated[dict[str, FundamentalSignal], _merge_dicts]
    news_signals: Annotated[dict[str, NewsSignal], _merge_dicts]
    decisions: Annotated[list[TradeDecision], operator.add]
    cycle_timestamp: str
