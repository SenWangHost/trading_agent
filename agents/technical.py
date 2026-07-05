import logging
import os
from datetime import datetime, timedelta

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
        current_price = float(df.iloc[-1]["close"])
        summary = _compute_indicators(ticker, df)
        llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(TechnicalSignal)
        signal = llm.invoke(
            f"Analyze these technical indicators for {ticker} and return a structured signal.\n\n{summary}"
        )
        log.info("%s technical → done | %s confidence=%.2f rsi=%.1f",
                 ticker, signal.direction, signal.confidence, signal.rsi)
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
    return {"technical_signals": {ticker: signal}, "prices": {ticker: current_price}}


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


def _compute_indicators(ticker: str, df: pd.DataFrame) -> str:
    df.ta.rsi(close="close", append=True)
    df.ta.macd(close="close", append=True)
    df.ta.sma(close="close", length=20, append=True)
    df.ta.sma(close="close", length=50, append=True)
    df.ta.atr(high="high", low="low", close="close", length=14, append=True)

    latest = df.iloc[-1]
    price = float(latest["close"])
    rsi = float(latest.get("RSI_14", 50))
    macd_line = float(latest.get("MACD_12_26_9", 0))
    macd_sig = float(latest.get("MACDs_12_26_9", 0))
    sma20 = float(latest.get("SMA_20", price))
    sma50 = float(latest.get("SMA_50", price))

    atr_col = next((c for c in df.columns if c.startswith("ATR")), None)
    atr = float(df[atr_col].iloc[-1]) if atr_col and not pd.isna(df[atr_col].iloc[-1]) else 0.0

    avg_vol = float(df["volume"].tail(20).mean())
    last_vol = float(latest["volume"])
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

    return (
        f"Ticker: {ticker} (Daily bars — swing trading context)\n"
        f"Current price: {price:.2f}\n"
        f"RSI(14): {rsi:.2f}\n"
        f"MACD line: {macd_line:.4f}, Signal line: {macd_sig:.4f} "
        f"({'bullish' if macd_line > macd_sig else 'bearish'} crossover)\n"
        f"SMA20: {sma20:.2f} ({'price above' if price > sma20 else 'price below'})\n"
        f"SMA50: {sma50:.2f} ({'price above' if price > sma50 else 'price below'})\n"
        f"ATR(14): {atr:.2f} (daily range volatility — use for stop-loss sizing: 1.5–2x ATR below entry)\n"
        f"Volume: {last_vol:,.0f} ({vol_ratio:.1f}x 20-day avg — high vol on breakouts is bullish confirmation)"
    )
