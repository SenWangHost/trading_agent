import pandas as pd
import pandas_ta as ta
import robin_stocks.robinhood as rh
from langchain_anthropic import ChatAnthropic

from state import AgentState, TechnicalSignal


def run_technical(state: AgentState) -> dict:
    ticker = state["current_ticker"]
    try:
        historicals = rh.stocks.get_stock_historicals(
            ticker, interval="5minute", span="week", bounds="regular"
        )
        df = _to_dataframe(historicals)
        current_price = float(df.iloc[-1]["close_price"])
        summary = _compute_indicators(ticker, df)
        llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(TechnicalSignal)
        signal = llm.invoke(
            f"Analyze these technical indicators for {ticker} and return a structured signal.\n\n{summary}"
        )
    except Exception as e:
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


def _to_dataframe(historicals: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(historicals)
    for col in ["close_price", "open_price", "high_price", "low_price", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df


def _compute_indicators(ticker: str, df: pd.DataFrame) -> str:
    df.ta.rsi(close="close_price", append=True)
    df.ta.macd(close="close_price", append=True)
    df.ta.sma(close="close_price", length=20, append=True)
    df.ta.sma(close="close_price", length=50, append=True)

    latest = df.iloc[-1]
    price = float(latest["close_price"])
    rsi = float(latest.get("RSI_14", 50))
    macd_line = float(latest.get("MACD_12_26_9", 0))
    macd_sig = float(latest.get("MACDs_12_26_9", 0))
    sma20 = float(latest.get("SMA_20", price))
    sma50 = float(latest.get("SMA_50", price))

    return (
        f"Ticker: {ticker}\n"
        f"Current price: {price:.2f}\n"
        f"RSI(14): {rsi:.2f}\n"
        f"MACD line: {macd_line:.4f}, Signal line: {macd_sig:.4f} "
        f"({'bullish' if macd_line > macd_sig else 'bearish'} crossover)\n"
        f"SMA20: {sma20:.2f} ({'price above' if price > sma20 else 'price below'})\n"
        f"SMA50: {sma50:.2f} ({'price above' if price > sma50 else 'price below'})"
    )
