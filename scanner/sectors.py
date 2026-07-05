# S&P 500 large-cap constituents organized by GICS sector.
# Edit these lists to add/remove tickers as index composition changes.

SECTOR_UNIVERSE: dict[str, list[str]] = {
    "technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "ORCL", "CRM", "CSCO",
        "IBM", "INTC", "TXN", "QCOM", "ACN", "NOW", "ADBE", "MU",
        "AMAT", "LRCX", "KLAC", "ADI", "MRVL", "ON", "CDNS", "SNPS", "ANSS",
    ],
    "semiconductors": [
        "NVDA", "AMD", "INTC", "TXN", "QCOM", "MU", "AVGO",
        "AMAT", "LRCX", "KLAC", "ADI", "MRVL", "ON", "CDNS", "SNPS",
        "MPWR", "SWKS", "QRVO", "MTSI", "WOLF",
    ],
    "communication_services": [
        "META", "GOOGL", "GOOG", "NFLX", "DIS", "TMUS", "VZ", "T",
        "CHTR", "EA", "TTWO", "WBD", "FOXA", "IPG", "OMC", "LYV",
        "MTCH", "RBLX", "SPOT", "SNAP",
    ],
    "consumer_discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX",
        "BKNG", "CMG", "ORLY", "AZO", "GM", "F", "CCL", "MAR",
        "HLT", "ROST", "DHI", "LEN", "PHM", "RCL", "NCLH", "EXPE", "MGM",
    ],
    "consumer_staples": [
        "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ",
        "CL", "GIS", "KHC", "CAG", "CPB", "CHD", "CLX", "KMB",
        "SYY", "HSY", "TSN", "HRL", "MKC",
    ],
    "healthcare": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR",
        "BMY", "AMGN", "ISRG", "SYK", "MDT", "BSX", "HUM", "CVS",
        "CI", "ELV", "MCK", "VRTX", "REGN", "GILD", "BIIB", "MRNA", "ZTS",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C",
        "SPGI", "MCO", "ICE", "CME", "CB", "PGR", "TRV", "AFL",
        "MET", "PRU", "AIG", "SCHW", "USB", "PNC", "TFC", "FITB", "COF",
    ],
    "energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
        "OXY", "HES", "DVN", "BKR", "HAL", "APA", "MRO", "CTRA",
        "TRGP", "KMI", "WMB",
    ],
    "materials": [
        "LIN", "APD", "SHW", "FCX", "NEM", "CTVA", "MOS", "CF",
        "EMN", "PPG", "ECL", "DD", "DOW", "ALB", "CE", "IFF",
        "VMC", "MLM", "RPM", "NUE", "STLD", "RS",
    ],
    "industrials": [
        "RTX", "HON", "UPS", "BA", "CAT", "DE", "LMT", "GE",
        "MMM", "FDX", "CSX", "UNP", "NSC", "ETN", "EMR", "PH",
        "ITW", "ROK", "GD", "NOC", "LHX", "TDG", "CARR", "OTIS", "CTAS", "FAST", "CPRT",
    ],
    "utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
        "ED", "WEC", "ES", "ETR", "FE", "PPL", "CMS", "AEE",
        "LNT", "PNW", "EVRG", "NI", "ATO", "AWK",
    ],
    "real_estate": [
        "PLD", "AMT", "EQIX", "CCI", "SPG", "O", "DLR", "PSA",
        "WELL", "AVB", "EQR", "VTR", "ARE", "BXP", "KIM", "REG",
        "UDR", "CPT", "ESS", "EXR", "CUBE", "INVH", "NNN",
    ],
}

# Convenience aliases → canonical sector key
_ALIASES: dict[str, str] = {
    "tech": "technology",
    "it": "technology",
    "information_technology": "technology",
    "software": "technology",
    "semi": "semiconductors",
    "semis": "semiconductors",
    "semiconductor": "semiconductors",
    "chips": "semiconductors",
    "comms": "communication_services",
    "communication": "communication_services",
    "communications": "communication_services",
    "media": "communication_services",
    "discretionary": "consumer_discretionary",
    "retail": "consumer_discretionary",
    "staples": "consumer_staples",
    "consumer": "consumer_staples",
    "health": "healthcare",
    "pharma": "healthcare",
    "biotech": "healthcare",
    "finance": "financials",
    "financial": "financials",
    "banks": "financials",
    "insurance": "financials",
    "oil": "energy",
    "gas": "energy",
    "material": "materials",
    "mining": "materials",
    "chemicals": "materials",
    "industrial": "industrials",
    "aerospace": "industrials",
    "defense": "industrials",
    "utility": "utilities",
    "power": "utilities",
    "reit": "real_estate",
    "reits": "real_estate",
    "realestate": "real_estate",
    "property": "real_estate",
}


def get_sector_universe(sectors: list[str]) -> set[str]:
    """
    Return the combined set of tickers for the given sector names.
    Accepts canonical names (e.g. "technology") or aliases (e.g. "tech").
    Unrecognized names are silently skipped.
    """
    tickers: set[str] = set()
    for raw in sectors:
        key = raw.lower().strip().replace(" ", "_").replace("-", "_")
        canonical = _ALIASES.get(key, key)
        tickers.update(SECTOR_UNIVERSE.get(canonical, []))
    return tickers


def list_sectors() -> list[str]:
    """Return the canonical sector names, sorted."""
    return sorted(SECTOR_UNIVERSE)
