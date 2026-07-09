import logging
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_ta  # registers df.ta accessor
from langchain_anthropic import ChatAnthropic

import polygon_client

from state import AgentState, TechnicalSignal

_POLYGON_BASE = "https://api.polygon.io"
log = logging.getLogger(__name__)


def run_technical(state: AgentState) -> dict:
    ticker = state["current_ticker"]
    log.info("%s technical → start", ticker)
    try:
        df = _fetch_bars(ticker)
        prev_close = float(df.iloc[-1]["close"])
        live_price = _fetch_snapshot_price(ticker, fallback=prev_close)
        summary = _compute_indicators(ticker, df, live_price)
        llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(TechnicalSignal)
        signal = llm.invoke(
            f"Analyze these technical indicators for {ticker} and return a structured signal.\n\n{summary}"
        )
        log.info("%s technical → done | %s confidence=%.2f rsi=%.1f price=%.2f",
                 ticker, signal.direction, signal.confidence, signal.rsi, live_price)
    except Exception as e:
        log.warning("%s technical → failed: %s", ticker, e)
        return {
            "technical_signals": {
                ticker: TechnicalSignal(
                    ticker=ticker,
                    direction="neutral",
                    confidence=0.0,
                    rsi=50.0,
                    macd_signal="unknown",
                    current_price=0.0,
                    reasoning=f"Data fetch failed: {e}",
                )
            },
            "prices": {ticker: 0.0},
        }
    return {"technical_signals": {ticker: signal}, "prices": {ticker: live_price}}


def _market_is_open() -> bool:
    """True if current ET time is within regular session hours on a weekday."""
    now = datetime.now(tz=ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    t = now.time()
    import datetime as dt
    return dt.time(9, 30) <= t <= dt.time(16, 0)


def _fetch_snapshot_price(ticker: str, fallback: float) -> float:
    """Return the latest minute-bar close from Polygon (includes pre-market and
    after-hours sessions); fall back to prev daily bar close on error/no data.

    On the Polygon Starter tier this price is ~15 min delayed. The 4-day window
    covers weekends/holidays so we still get the most recent extended-hours bar.
    """
    market_open = _market_is_open()
    api_key = os.environ["POLYGON_API_KEY"]
    end = datetime.now(tz=ZoneInfo("America/New_York"))
    start = end - timedelta(days=4)
    try:
        resp = polygon_client.get(
            f"{_POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/minute"
            f"/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
            params={"adjusted": "true", "sort": "desc", "limit": 1, "apiKey": api_key},
            timeout=10,
        )
        results = resp.json().get("results", [])
        if results:
            price = float(results[0]["c"])
            log.info("%s → price: %.2f (Polygon minute bar, market_open=%s)", ticker, price, market_open)
            return price
        log.warning("%s Polygon returned no minute bar, falling back to prev close: %.2f", ticker, fallback)
    except Exception as e:
        log.warning("%s Polygon price fetch failed, falling back to prev close %.2f: %s", ticker, fallback, e)
    return fallback


def _fetch_bars(ticker: str) -> pd.DataFrame:
    api_key = os.environ["POLYGON_API_KEY"]
    end = datetime.now()
    start = end - timedelta(days=90)
    resp = polygon_client.get(
        f"{_POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day"
        f"/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
        params={"adjusted": "true", "sort": "asc", "limit": 200, "apiKey": api_key},
        timeout=10,
    )
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"No bar data returned for {ticker}")
    df = pd.DataFrame(results).rename(columns={
        "c": "close", "o": "open", "h": "high", "l": "low", "v": "volume", "t": "timestamp",
    })
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _compute_indicators(ticker: str, df: pd.DataFrame, live_price: float) -> str:
    df.ta.rsi(close="close", append=True)
    df.ta.macd(close="close", append=True)
    df.ta.sma(close="close", length=20, append=True)
    df.ta.sma(close="close", length=50, append=True)
    df.ta.atr(high="high", low="low", close="close", length=14, append=True)

    latest = df.iloc[-1]
    prev_close = float(latest["close"])
    rsi = float(latest.get("RSI_14", 50))
    macd_line = float(latest.get("MACD_12_26_9", 0))
    macd_sig = float(latest.get("MACDs_12_26_9", 0))
    sma20 = float(latest.get("SMA_20", prev_close))
    sma50 = float(latest.get("SMA_50", prev_close))

    atr_col = next((c for c in df.columns if c.startswith("ATR")), None)
    atr = float(df[atr_col].iloc[-1]) if atr_col and not pd.isna(df[atr_col].iloc[-1]) else 0.0

    avg_vol = float(df["volume"].tail(20).mean())
    last_vol = float(latest["volume"])
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

    market_open = _market_is_open()
    price_label = "Live price (market open)" if market_open else "Last session close (market closed)"
    return (
        f"Ticker: {ticker} (Daily bars — swing trading context)\n"
        f"{price_label}: {live_price:.2f}\n"
        f"Prev daily bar close: {prev_close:.2f} (basis for all indicators below)\n"
        f"RSI(14): {rsi:.2f}\n"
        f"MACD line: {macd_line:.4f}, Signal line: {macd_sig:.4f} "
        f"({'bullish' if macd_line > macd_sig else 'bearish'} crossover)\n"
        f"SMA20: {sma20:.2f} ({'price above' if live_price > sma20 else 'price below'})\n"
        f"SMA50: {sma50:.2f} ({'price above' if live_price > sma50 else 'price below'})\n"
        f"ATR(14): {atr:.2f} (daily range volatility — use for stop-loss sizing: 1.5–2x ATR below entry)\n"
        f"Volume: {last_vol:,.0f} ({vol_ratio:.1f}x 20-day avg — high vol on breakouts is bullish confirmation)"
    )
