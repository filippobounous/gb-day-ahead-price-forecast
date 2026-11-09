"""One shared HTTP session with automatic retries for every data source."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "gb-da-forecast/0.1 (open-data research script)"

_session: requests.Session | None = None


def session() -> requests.Session:
    """Return a session that retries rate limits (429) and server errors with exponential backoff."""
    global _session
    if _session is None:
        retry = Retry(
            total=6,
            backoff_factor=1.0,  # waits 0, 2, 4, 8, 16, 32 s between attempts; honours Retry-After
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
        s.headers["User-Agent"] = USER_AGENT
        _session = s
    return _session


def get_json(url: str, params: dict | None = None, timeout: float = 120):
    response = session().get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_bytes(url: str, params: dict | None = None, timeout: float = 120) -> bytes:
    response = session().get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.content
