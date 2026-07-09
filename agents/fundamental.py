import datetime
import logging
import os

import polygon_client
import yfinance as yf
from langchain_anthropic import ChatAnthropic

from state import AgentState, FundamentalSignal

_POLYGON_BASE = "https://api.polygon.io"
_cache: dict[str, tuple[str, FundamentalSignal]] = {}
log = logging.getLogger(__name__)


def run_fundamental(state: AgentState) -> dict:
    ticker = state["current_ticker"]
    today = datetime.date.today().isoformat()

    if ticker in _cache and _cache[ticker][0] == today:
        log.info("%s fundamental → cache hit", ticker)
        return {"fundamental_signals": {ticker: _cache[ticker][1]}}

    log.info("%s fundamental → start", ticker)
    try:
        summary = _fetch_and_format(ticker) + "\n" + _earnings_calendar(ticker)
        llm = ChatAnthropic(model="claude-sonnet-4-6").with_structured_output(FundamentalSignal)
        signal = llm.invoke(
            f"Analyze these fundamental metrics for {ticker} and return a structured signal.\n"
            f"Note: pe_ratio and pb_ratio require current price which is not available here — set both to null. "
            f"week_52_high and week_52_low are not in this data source — set both to null.\n"
            f"Important: if earnings are within 7 days, reduce confidence to reflect event risk.\n\n{summary}"
        )
        log.info("%s fundamental → done | %s confidence=%.2f", ticker, signal.direction, signal.confidence)
    except Exception as e:
        log.warning("%s fundamental → failed: %s", ticker, e)
        signal = FundamentalSignal(
            ticker=ticker,
            direction="neutral",
            confidence=0.0,
            pe_ratio=None,
            pb_ratio=None,
            market_cap=None,
            week_52_high=None,
            week_52_low=None,
            reasoning=f"Data fetch failed: {e}",
        )

    _cache[ticker] = (today, signal)
    return {"fundamental_signals": {ticker: signal}}


def _fetch_and_format(ticker: str) -> str:
    api_key = os.environ["POLYGON_API_KEY"]

    details_resp = polygon_client.get(
        f"{_POLYGON_BASE}/v3/reference/tickers/{ticker}",
        params={"apiKey": api_key},
        timeout=10,
    )
    details = details_resp.json().get("results", {})

    fin_resp = polygon_client.get(
        f"{_POLYGON_BASE}/vX/reference/financials",
        params={"ticker": ticker, "timeframe": "quarterly", "limit": 4, "apiKey": api_key},
        timeout=10,
    )
    financials = fin_resp.json().get("results", [])

    return _format(ticker, details, financials)


def _earnings_calendar(ticker: str) -> str:
    try:
        cal = yf.Ticker(ticker).calendar
        if not cal:
            return "Next earnings date: unknown"

        dates = cal.get("Earnings Date", [])
        if dates:
            next_date = dates[0]
            days_until = (next_date - datetime.date.today()).days
            date_str = str(next_date)
            timing = f"{days_until} days from now"
            if days_until <= 7:
                timing += " — WARNING: earnings within 7 days, elevated event risk"
        else:
            date_str = "unknown"
            timing = ""

        eps_avg = cal.get("Earnings Average")
        eps_low = cal.get("Earnings Low")
        eps_high = cal.get("Earnings High")
        rev_avg = cal.get("Revenue Average")

        lines = ["", "--- Earnings Calendar ---"]
        lines.append(f"Next earnings date: {date_str}" + (f" ({timing})" if timing else ""))
        if eps_avg is not None:
            lines.append(f"EPS estimate: avg ${eps_avg:.2f} (range ${eps_low:.2f}–${eps_high:.2f})")
        if rev_avg is not None:
            lines.append(f"Revenue estimate: {_fmt_num(rev_avg)}")
        return "\n".join(lines)
    except Exception:
        return "\nNext earnings date: unavailable"


def _format(ticker: str, details: dict, financials: list) -> str:
    market_cap = details.get("market_cap")
    description = str(details.get("description", ""))[:400]
    shares_outstanding = details.get("share_class_shares_outstanding")

    ttm_revenue = ttm_net_income = ttm_eps = 0.0
    equity = assets = None

    for q in financials:
        fin = q.get("financials", {})
        inc = fin.get("income_statement", {})
        bal = fin.get("balance_sheet", {})

        ttm_revenue += _val(inc, "revenues")
        ttm_net_income += _val(inc, "net_income_loss")
        ttm_eps += _val(inc, "basic_earnings_per_share")

        if equity is None and bal.get("equity"):
            equity = _val(bal, "equity")
        if assets is None and bal.get("assets"):
            assets = _val(bal, "assets")

    lines = [
        f"Ticker: {ticker}",
        f"Market cap: {_fmt_num(market_cap)}",
        f"Shares outstanding: {_fmt_num(shares_outstanding)}",
        f"TTM Revenue: {_fmt_num(ttm_revenue)}",
        f"TTM Net Income: {_fmt_num(ttm_net_income)}",
        f"TTM EPS (basic): {ttm_eps:.4f}" if ttm_eps else "TTM EPS: N/A",
        f"Total Equity (latest quarter): {_fmt_num(equity)}",
        f"Total Assets (latest quarter): {_fmt_num(assets)}",
        f"Description: {description}",
    ]
    return "\n".join(l for l in lines if l)


def _val(d: dict, key: str) -> float:
    item = d.get(key, {})
    if isinstance(item, dict):
        return float(item.get("value", 0) or 0)
    return 0.0


def _fmt_num(val) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if v >= 1e12:
            return f"${v / 1e12:.2f}T"
        if v >= 1e9:
            return f"${v / 1e9:.2f}B"
        if v >= 1e6:
            return f"${v / 1e6:.2f}M"
        return f"${v:.2f}"
    except (TypeError, ValueError):
        return "N/A"
