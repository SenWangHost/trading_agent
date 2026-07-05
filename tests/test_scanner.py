import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("POLYGON_API_KEY", "test_polygon_key")


def _grouped_bars_response():
    return {
        "status": "OK",
        "results": [
            {"T": "AAPL", "c": 195.0, "o": 193.0, "v": 80_000_000},
            {"T": "NVDA", "c": 130.0, "o": 128.0, "v": 60_000_000},
            {"T": "TSLA", "c": 250.0, "o": 248.0, "v": 45_000_000},
            {"T": "MSFT", "c": 420.0, "o": 418.0, "v": 25_000_000},
            {"T": "PENNY", "c": 3.0,   "o": 2.9,   "v": 5_000_000},   # < min_price
            {"T": "ILLIQ", "c": 50.0,  "o": 49.0,  "v": 100_000},     # < min_volume
            {"T": "AAPL231215C00200000", "c": 5.0, "o": 4.9, "v": 50_000_000},  # non-alpha — filtered
        ],
    }


def _mock_get(url, **kwargs):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = _grouped_bars_response()
    return resp


def test_scan_candidates_filters_price_and_volume():
    from scanner.polygon import scan_candidates

    with patch("polygon_client.requests.get", side_effect=_mock_get):
        results = scan_candidates(min_price=10.0, min_volume=500_000, max_results=20)

    assert "AAPL" in results
    assert "NVDA" in results
    assert "TSLA" in results
    assert "MSFT" in results
    assert "PENNY" not in results   # below min_price
    assert "ILLIQ" not in results   # below min_volume


def test_scan_candidates_sorted_by_volume_descending():
    from scanner.polygon import scan_candidates

    with patch("polygon_client.requests.get", side_effect=_mock_get):
        results = scan_candidates(min_price=10.0, min_volume=500_000, max_results=20)

    # AAPL (80M) > NVDA (60M) > TSLA (45M) > MSFT (25M)
    assert results.index("AAPL") < results.index("NVDA")
    assert results.index("NVDA") < results.index("TSLA")
    assert results.index("TSLA") < results.index("MSFT")


def test_scan_candidates_respects_max_results():
    from scanner.polygon import scan_candidates

    with patch("polygon_client.requests.get", side_effect=_mock_get):
        results = scan_candidates(min_price=10.0, min_volume=500_000, max_results=2)

    assert len(results) == 2
    assert results[0] == "AAPL"  # highest volume
    assert results[1] == "NVDA"


def test_scan_candidates_excludes_non_alpha_tickers():
    from scanner.polygon import scan_candidates

    with patch("polygon_client.requests.get", side_effect=_mock_get):
        results = scan_candidates(min_price=1.0, min_volume=0, max_results=50)

    assert all(t.isalpha() for t in results)
    assert all(len(t) <= 5 for t in results)


def test_scan_candidates_returns_empty_on_api_failure():
    from scanner.polygon import scan_candidates

    with patch("polygon_client.requests.get", side_effect=Exception("network error")):
        results = scan_candidates()

    assert results == []


def test_scan_candidates_returns_empty_when_no_bars():
    from scanner.polygon import scan_candidates

    def _empty_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"status": "OK", "results": []}
        return resp

    with patch("polygon_client.requests.get", side_effect=_empty_get):
        results = scan_candidates()

    assert results == []


# --- sector filtering ---

def test_sector_filter_restricts_to_universe():
    from scanner.polygon import scan_candidates

    # Bars contain AAPL/NVDA/MSFT (technology) and TSLA (consumer_discretionary)
    with patch("polygon_client.requests.get", side_effect=_mock_get):
        results = scan_candidates(min_price=10.0, min_volume=500_000, max_results=20,
                                  sectors=["technology"])

    assert "AAPL" in results
    assert "NVDA" in results
    assert "MSFT" in results
    assert "TSLA" not in results   # consumer_discretionary, not technology


def test_sector_filter_multiple_sectors():
    from scanner.polygon import scan_candidates

    with patch("polygon_client.requests.get", side_effect=_mock_get):
        results = scan_candidates(min_price=10.0, min_volume=500_000, max_results=20,
                                  sectors=["technology", "consumer_discretionary"])

    assert "AAPL" in results
    assert "TSLA" in results


def test_sector_filter_alias_tech():
    from scanner.polygon import scan_candidates

    with patch("polygon_client.requests.get", side_effect=_mock_get):
        results_alias = scan_candidates(min_price=10.0, min_volume=500_000, max_results=20,
                                        sectors=["tech"])
        results_canonical = scan_candidates(min_price=10.0, min_volume=500_000, max_results=20,
                                            sectors=["technology"])

    assert results_alias == results_canonical


def test_sector_filter_unknown_sector_returns_empty():
    from scanner.polygon import scan_candidates

    with patch("polygon_client.requests.get", side_effect=_mock_get):
        results = scan_candidates(min_price=10.0, min_volume=500_000, max_results=20,
                                  sectors=["not_a_real_sector"])

    assert results == []


def test_sector_filter_empty_list_scans_all():
    from scanner.polygon import scan_candidates

    with patch("polygon_client.requests.get", side_effect=_mock_get):
        results_no_filter = scan_candidates(min_price=10.0, min_volume=500_000, max_results=20,
                                            sectors=[])
        results_none = scan_candidates(min_price=10.0, min_volume=500_000, max_results=20,
                                       sectors=None)

    assert results_no_filter == results_none


# --- sectors.py unit tests ---

def test_get_sector_universe_returns_tickers_for_known_sector():
    from scanner.sectors import get_sector_universe

    tickers = get_sector_universe(["technology"])
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "NVDA" in tickers


def test_get_sector_universe_alias_resolves():
    from scanner.sectors import get_sector_universe

    assert get_sector_universe(["tech"]) == get_sector_universe(["technology"])
    assert get_sector_universe(["pharma"]) == get_sector_universe(["healthcare"])
    assert get_sector_universe(["reit"]) == get_sector_universe(["real_estate"])
    assert get_sector_universe(["banks"]) == get_sector_universe(["financials"])


def test_get_sector_universe_combines_multiple_sectors():
    from scanner.sectors import get_sector_universe

    combined = get_sector_universe(["technology", "healthcare"])
    tech_only = get_sector_universe(["technology"])
    health_only = get_sector_universe(["healthcare"])

    assert combined == tech_only | health_only


def test_get_sector_universe_unknown_returns_empty():
    from scanner.sectors import get_sector_universe

    assert get_sector_universe(["not_a_sector"]) == set()
    assert get_sector_universe([]) == set()


def test_list_sectors_returns_all_canonical_names():
    from scanner.sectors import list_sectors, SECTOR_UNIVERSE

    sectors = list_sectors()
    assert set(sectors) == set(SECTOR_UNIVERSE.keys())
    assert sectors == sorted(sectors)
