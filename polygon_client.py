import logging
import random
import time

import requests

log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BASE_WAIT = 15  # seconds; Polygon free tier allows 5 req/min (≈12s per call)


def get(url: str, **kwargs) -> requests.Response:
    """
    requests.get with automatic retry on Polygon 429 rate-limit responses.
    Waits 15s, 30s, 45s (+jitter) between attempts before giving up.
    """
    for attempt in range(_MAX_ATTEMPTS):
        resp = requests.get(url, **kwargs)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = _BASE_WAIT * (attempt + 1) + random.uniform(0, 5)
        log.warning(
            "Polygon rate limit (429) — retrying in %.0fs (attempt %d/%d)",
            wait, attempt + 1, _MAX_ATTEMPTS,
        )
        time.sleep(wait)
    resp.raise_for_status()
    return resp
