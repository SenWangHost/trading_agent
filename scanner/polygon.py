import logging
import os
from datetime import date, timedelta

import polygon_client

from scanner.sectors import get_sector_universe

_POLYGON_BASE = "https://api.polygon.io"
log = logging.getLogger(__name__)


def scan_candidates(
    min_price: float | None = None,
    min_volume: int | None = None,
    max_results: int | None = None,
    sectors: list[str] | None = None,
) -> list[str]:
    """
    Fetch the most recent trading day's grouped bars from Polygon and return
    the top tickers by volume that pass the price/volume/sector filters.

    sectors — restrict candidates to these GICS sectors (canonical names or aliases).
              If None, reads SCANNER_SECTORS env var; if that is also unset, all
              liquid tickers are eligible.

    Other defaults are read from env vars: SCANNER_MIN_PRICE, SCANNER_MIN_VOLUME,
    SCANNER_MAX_RESULTS.
    """
    api_key = os.environ["POLYGON_API_KEY"]
    min_price = min_price if min_price is not None else float(os.environ.get("SCANNER_MIN_PRICE", "10"))
    min_volume = min_volume if min_volume is not None else int(os.environ.get("SCANNER_MIN_VOLUME", "1000000"))
    max_results = max_results if max_results is not None else int(os.environ.get("SCANNER_MAX_RESULTS", "10"))

    if sectors is None:
        sector_env = os.environ.get("SCANNER_SECTORS", "").strip()
        sectors = [s.strip() for s in sector_env.split(",") if s.strip()] if sector_env else []

    universe: set[str] | None = get_sector_universe(sectors) if sectors else None
    sector_label = ", ".join(sectors) if sectors else "all"
    log.info("scanner → start (sectors=%s min_price=%.0f min_volume=%d max=%d)",
             sector_label, min_price, min_volume, max_results)

    bars = _fetch_grouped_bars(api_key)
    if not bars:
        log.warning("scanner → no bars returned, skipping")
        return []

    candidates = [
        r for r in bars
        if _is_stock_ticker(r.get("T", ""))
        and (universe is None or r.get("T") in universe)
        and r.get("c", 0) >= min_price
        and r.get("v", 0) >= min_volume
    ]
    candidates.sort(key=lambda r: r.get("v", 0), reverse=True)
    results = [r["T"] for r in candidates[:max_results]]
    log.info("scanner → done | %d candidates: %s", len(results), ", ".join(results))
    return results


def _fetch_grouped_bars(api_key: str) -> list[dict]:
    """Try up to 5 calendar days back to find the most recent trading day."""
    for days_back in range(1, 6):
        target_date = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            resp = polygon_client.get(
                f"{_POLYGON_BASE}/v2/aggs/grouped/locale/us/market/stocks/{target_date}",
                params={"adjusted": "true", "apiKey": api_key},
                timeout=30,
            )
            results = resp.json().get("results", [])
            if results:
                return results
        except Exception:
            continue
    return []


def _is_stock_ticker(ticker: str) -> bool:
    """Accept only plain alphabetic tickers (1–5 chars). Excludes options, warrants, ADR suffixes."""
    return ticker.isalpha() and 1 <= len(ticker) <= 5
