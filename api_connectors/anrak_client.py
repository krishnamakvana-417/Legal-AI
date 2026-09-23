"""
AnrakLegal API Client
──────────────────────
Async httpx client for the AnrakLegal REST API.
Endpoints: eCourts case lookup, citator flags, statute search.
Responses cached with 60-minute TTL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

# pyrefly: ignore [missing-import]
import httpx
from cachetools import TTLCache

logger = logging.getLogger(__name__)

_BASE_URL = os.getenv("ANRAK_BASE_URL", "https://api.anraklegal.com")
_API_KEY = os.getenv("ANRAK_API_KEY", "")

_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# TTL cache: max 500 items, 60-minute TTL
_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _auth_headers() -> dict[str, str]:
    return {
        "X-API-Key": _API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


async def _get(
    url: str,
    params: Optional[dict] = None,
    cache_key: Optional[str] = None,
) -> dict[str, Any]:
    """GET with caching and exponential backoff retry."""
    # Cache check
    if cache_key and cache_key in _cache:
        logger.debug(f"Cache hit: {cache_key}")
        return _cache[cache_key]

    headers = _auth_headers()
    last_exc: Exception = RuntimeError("Unknown")

    for attempt, backoff in enumerate(_RETRY_BACKOFF, start=1):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                if cache_key:
                    _cache[cache_key] = data
                return data
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning(f"Anrak rate limited. Retrying in {backoff}s (attempt {attempt})")
                await asyncio.sleep(backoff)
                last_exc = exc
            elif exc.response.status_code in {401, 403}:
                raise PermissionError(f"AnrakLegal auth error: {exc.response.text}") from exc
            elif exc.response.status_code == 404:
                return {"error": "not_found", "status": 404}
            else:
                raise
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.warning(f"Anrak network error (attempt {attempt}): {exc}")
            await asyncio.sleep(backoff)
            last_exc = exc

    raise RuntimeError(f"AnrakLegal request failed after {_MAX_RETRIES} retries: {last_exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Client
# ═══════════════════════════════════════════════════════════════════════════════

class AnrakClient:
    """
    Async client for the AnrakLegal API.

    Usage:
        client = AnrakClient()
        case = await client.get_case_by_cnr("MHNS010123456")
        citator = await client.get_citator("doc123")
        statutes = await client.search_statutes("section 498A IPC")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _BASE_URL,
    ):
        self.api_key = api_key or _API_KEY
        self.base_url = base_url.rstrip("/")

        if not self.api_key:
            logger.warning(
                "AnrakLegal API key not configured. "
                "Set ANRAK_API_KEY in your .env file."
            )

    async def get_case_by_cnr(self, cnr: str) -> dict[str, Any]:
        """
        Fetch live eCourts case status by Case Number Reference (CNR).

        Args:
            cnr: eCourts CNR number (e.g. "MHNS010123456789").

        Returns:
            Case details dict: case_title, court, filing_date, status, next_hearing, etc.
        """
        url = f"{self.base_url}/api/v1/courts/cases/{cnr}"
        cache_key = f"cnr:{cnr}"
        logger.debug(f"Anrak case lookup: CNR={cnr}")
        return await _get(url, cache_key=cache_key)

    async def get_citator(self, doc_id: str) -> dict[str, Any]:
        """
        Fetch citator flags for a document (positive/negative treatment history).

        Args:
            doc_id: Internal document ID.

        Returns:
            Dict with treatment flags: overruled, distinguished, followed, affirmed, etc.
        """
        url = f"{self.base_url}/api/v1/citator/{doc_id}"
        cache_key = f"citator:{doc_id}"
        logger.debug(f"Anrak citator: doc_id={doc_id}")
        return await _get(url, cache_key=cache_key)

    async def search_statutes(
        self,
        query: str,
        act: Optional[str] = None,
        section: Optional[str] = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Search for statutory provisions.

        Args:
            query: Free-text search query.
            act: Restrict to a specific act name (e.g. "Indian Penal Code").
            section: Restrict to a specific section number.
            limit: Maximum results to return.

        Returns:
            Dict with 'results' list of matching statute sections.
        """
        url = f"{self.base_url}/api/v1/statutes/search"
        params: dict[str, Any] = {"q": query, "limit": limit}
        if act:
            params["act"] = act
        if section:
            params["section"] = section

        cache_key = f"statute:{query}:{act}:{section}:{limit}"
        logger.debug(f"Anrak statute search: q='{query}'")
        return await _get(url, params=params, cache_key=cache_key)

    async def get_statute_section(self, act_id: str, section_id: str) -> dict[str, Any]:
        """
        Fetch the full text of a specific statutory section.

        Args:
            act_id: Act identifier slug.
            section_id: Section identifier.

        Returns:
            Dict with section_number, section_title, text, act_name, etc.
        """
        url = f"{self.base_url}/api/v1/statutes/{act_id}/sections/{section_id}"
        cache_key = f"section:{act_id}:{section_id}"
        return await _get(url, cache_key=cache_key)

    def clear_cache(self) -> None:
        """Clear the response cache."""
        _cache.clear()
        logger.info("AnrakLegal response cache cleared.")


# Singleton convenience instance
anrak = AnrakClient()
