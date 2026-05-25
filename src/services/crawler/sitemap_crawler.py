"""Sitemap-based website crawler."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import requests

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Playwright

LOGGER = logging.getLogger(__name__)


class SitemapCrawler:
    """Fetch URLs from sitemap.xml and render page HTML via Playwright."""

    def __init__(
        self,
        base_url: str,
        user_agent: str,
        timeout: int,
        manual_urls: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._manual_urls = manual_urls
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None

    def discover_urls(self) -> list[str]:
        """Resolve URLs from a sitemap or sitemap index.

        When manual_urls are configured they take priority and sitemap
        discovery is skipped entirely. Sitemap discovery is only used
        as a fallback when no manual_urls are provided.
        """

        if self._manual_urls.strip():
            manual_urls = self._discover_manual_urls()
            LOGGER.info(
                "URL discovery mode: manual; discovered %s URLs",
                len(manual_urls),
            )
            return manual_urls

        sitemap_urls = self._discover_sitemap_urls()
        if sitemap_urls:
            LOGGER.info(
                "URL discovery mode: sitemap; discovered %s URLs",
                len(sitemap_urls),
            )
            return sitemap_urls

        raise ValueError(
            "No valid sitemap URLs found and WEBSITE_URLS is not configured."
        )

    def fetch_page(self, url: str) -> str:
        """Render a page with Playwright and return the full HTML."""

        from playwright.sync_api import (  # noqa: PLC0415
            TimeoutError as PlaywrightTimeoutError,
            sync_playwright,
        )

        if self._playwright is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)

        assert self._browser is not None
        page = self._browser.new_page()
        try:
            page.goto(
                url,
                timeout=self._timeout * 1000,
                wait_until="networkidle",
            )
            html = page.content()
            LOGGER.info("Rendered page at %s: %d chars", url, len(html))
            return html
        except PlaywrightTimeoutError:
            LOGGER.warning(
                "Playwright timed out after %ds for %s; retrying with domcontentloaded",
                self._timeout,
                url,
            )
            page.goto(url, timeout=self._timeout * 1000, wait_until="domcontentloaded")
            html = page.content()
            LOGGER.info("Rendered page at %s: %d chars (fallback)", url, len(html))
            return html
        finally:
            page.close()

    def close(self) -> None:
        """Release Playwright browser resources."""

        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def __del__(self) -> None:
        self.close()

    def _discover_sitemap_urls(self) -> list[str]:
        sitemap_url = urljoin(f"{self._base_url}/", "sitemap.xml")
        try:
            response = self._session.get(sitemap_url, timeout=self._timeout)
            response.raise_for_status()
            root = ElementTree.fromstring(response.text)
        except (ElementTree.ParseError, requests.RequestException) as exc:
            LOGGER.warning("Sitemap URL discovery failed: %s", exc)
            return []

        namespace = self._extract_namespace(root.tag)
        discovered = self._parse_sitemap(root, namespace)
        return self._deduplicate_urls(discovered)

    def _discover_manual_urls(self) -> list[str]:
        if not self._manual_urls.strip():
            return []

        urls = []
        for raw_url in self._manual_urls.split(","):
            url = raw_url.strip()
            if not url:
                continue

            urls.append(self._to_absolute_url(url))

        return self._deduplicate_urls(urls)

    def _parse_sitemap(self, root: ElementTree.Element, namespace: str) -> list[str]:
        url_tag = f".//{namespace}url/{namespace}loc"
        sitemap_tag = f".//{namespace}sitemap/{namespace}loc"

        urls = [node.text.strip() for node in root.findall(url_tag) if node.text]
        nested_sitemaps = [
            node.text.strip()
            for node in root.findall(sitemap_tag)
            if node.text
        ]

        if urls:
            domain = urlparse(self._base_url).netloc
            return [url for url in urls if urlparse(url).netloc == domain]

        nested_urls: list[str] = []
        for sitemap_url in nested_sitemaps:
            try:
                response = self._session.get(sitemap_url, timeout=self._timeout)
                response.raise_for_status()
                nested_root = ElementTree.fromstring(response.text)
            except (ElementTree.ParseError, requests.RequestException) as exc:
                LOGGER.warning("Nested sitemap URL discovery failed: %s", exc)
                continue

            nested_namespace = self._extract_namespace(nested_root.tag)
            nested_urls.extend(
                self._parse_sitemap(nested_root, nested_namespace)
            )
        return nested_urls

    def _to_absolute_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            return url

        return urljoin(f"{self._base_url}/", url)

    @staticmethod
    def _deduplicate_urls(urls: list[str]) -> list[str]:
        return sorted(set(urls))

    @staticmethod
    def _extract_namespace(tag: str) -> str:
        if tag.startswith("{"):
            return tag.split("}")[0] + "}"
        return ""
