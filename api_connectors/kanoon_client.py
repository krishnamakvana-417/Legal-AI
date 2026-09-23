"""
Indian Kanoon API Client
─────────────────────────
Async httpx client for the Indian Kanoon REST API.
Supports Token-based auth and HMAC public-private key auth.
Endpoints: /search/, /doc/{docid}/, /docfragment/{docid}/
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from typing import Any, Optional
from urllib.parse import urlencode

# pyrefly: ignore [missing-import]
import httpx

logger = logging.getLogger(__name__)

_BASE_URL = os.getenv("INDIAN_KANOON_BASE_URL", "https://api.indiankanoon.org")
_API_TOKEN = os.getenv("INDIAN_KANOON_API_TOKEN", "")

# Retry config
_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]  # seconds
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Auth Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _token_headers() -> dict[str, str]:
    """Standard API token auth headers."""
    return {
        "Authorization": f"Token {_API_TOKEN}",
        "Accept": "application/json",
    }


def _hmac_headers(
    method: str,
    path: str,
    public_key: str,
    private_key: str,
) -> dict[str, str]:
    """
    Generate HMAC-SHA256 authentication headers.
    Signature = HMAC-SHA256(private_key, "{method}\n{path}\n{timestamp}")
    """
    timestamp = str(int(time.time()))
    message = f"{method.upper()}\n{path}\n{timestamp}"
    sig = hmac.new(
        private_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {
        "X-Public-Key": public_key,
        "X-Timestamp": timestamp,
        "X-Signature": sig,
        "Accept": "application/json",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Helper with Retry
# ═══════════════════════════════════════════════════════════════════════════════

async def _get(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> dict[str, Any]:
    """Async GET with exponential backoff retry."""
    import asyncio

    last_exc: Exception = RuntimeError("Unknown error")
    for attempt, backoff in enumerate(_RETRY_BACKOFF, start=1):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning(f"Rate limited. Waiting {backoff}s (attempt {attempt}/{_MAX_RETRIES})")
                await asyncio.sleep(backoff)
                last_exc = exc
            elif exc.response.status_code in {401, 403}:
                raise PermissionError(f"Kanoon API auth error: {exc.response.status_code}") from exc
            else:
                raise
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.warning(f"Network error on attempt {attempt}: {exc}")
            await asyncio.sleep(backoff)
            last_exc = exc

    raise RuntimeError(f"Kanoon API request failed after {_MAX_RETRIES} retries: {last_exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# API Methods
# ═══════════════════════════════════════════════════════════════════════════════

class KanoonClient:
    """
    Async client for the Indian Kanoon REST API.

    Usage:
        client = KanoonClient()
        results = await client.search("right to bail CrPC 439", pagenum=0)
        doc = await client.get_document(docid=1234567)
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        public_key: Optional[str] = None,
        private_key: Optional[str] = None,
        base_url: str = _BASE_URL,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token or _API_TOKEN
        self.public_key = public_key
        self.private_key = private_key

        if not self.api_token and not (self.public_key and self.private_key):
            logger.warning(
                "No Indian Kanoon credentials configured. "
                "Set INDIAN_KANOON_API_TOKEN or provide public/private keys."
            )

    def _headers(self, method: str = "GET", path: str = "/") -> dict[str, str]:
        if self.public_key and self.private_key:
            return _hmac_headers(method, path, self.public_key, self.private_key)
        return _token_headers()

    async def search(
        self,
        query: str,
        pagenum: int = 0,
        doc_types: Optional[list[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Search Indian Kanoon.

        Args:
            query: Search query string.
            pagenum: Page number (0-indexed, ~10 results per page).
            doc_types: List of document types, e.g. ["judgments", "acts"].
            from_date: Filter from date (DD-MM-YYYY).
            to_date: Filter to date (DD-MM-YYYY).

        Returns:
            Raw Kanoon API response dict with 'docs' key.
        """
        path = "/search/"
        params: dict[str, Any] = {"formInput": query, "pagenum": pagenum}
        if doc_types:
            params["doctype"] = ",".join(doc_types)
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        url = self.base_url + path
        headers = self._headers("GET", path)

        logger.debug(f"Kanoon search: q='{query}' page={pagenum}")
        return await _get(url, params=params, headers=headers)

    async def get_document(self, docid: int | str) -> dict[str, Any]:
        """
        Fetch a full judgment document.

        Args:
            docid: Indian Kanoon document ID.

        Returns:
            Dict with 'doc' (HTML text) and metadata fields.
        """
        path = f"/doc/{docid}/"
        url = self.base_url + path
        headers = self._headers("GET", path)

        logger.debug(f"Kanoon get doc: {docid}")
        return await _get(url, headers=headers)

    async def get_doc_fragment(
        self,
        docid: int | str,
        query: str,
        fragment_count: int = 5,
    ) -> dict[str, Any]:
        """
        Fetch highlighted document fragments matching a query.

        Args:
            docid: Indian Kanoon document ID.
            query: Query to highlight in the document.
            fragment_count: Number of highlighted passages to return.

        Returns:
            Dict with 'fragments' key containing highlighted snippets.
        """
        path = f"/docfragment/{docid}/"
        params = {"formInput": query, "fragmentCount": fragment_count}
        url = self.base_url + path
        headers = self._headers("GET", path)

        logger.debug(f"Kanoon get fragment: doc={docid} q='{query}'")
        return await _get(url, params=params, headers=headers)

    async def search_all_pages(
        self,
        query: str,
        max_pages: int = 5,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Paginate through multiple search result pages.

        Args:
            query: Search query.
            max_pages: Maximum pages to fetch.
            **kwargs: Additional params forwarded to search().

        Returns:
            Flat list of document result dicts.
        """
        all_docs: list[dict] = []
        for page in range(max_pages):
            try:
                data = await self.search(query, pagenum=page, **kwargs)
                docs = data.get("docs", [])
                if not docs:
                    break
                all_docs.extend(docs)
                logger.debug(f"  Page {page}: {len(docs)} docs (total: {len(all_docs)})")
            except Exception as exc:
                logger.warning(f"Page {page} fetch error: {exc}")
                break
        return all_docs


# Singleton convenience instance
kanoon = KanoonClient()
