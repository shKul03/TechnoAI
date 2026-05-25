import sys
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

"""Standalone ingestion worker.

Run directly with the project venv:
    python -m src.services.ingestion.ingest_worker
    -- or --
    python src/services/ingestion/ingest_worker.py

The ProactorEventLoop policy is set as the very first executable
statement so that Playwright can spawn its browser subprocess.
This file must NOT be imported by the FastAPI process (which needs
the SelectorEventLoop for psycopg async).
"""

import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

# Ensure project root is importable when run as a plain script
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    from src.config.settings import get_settings
    from src.services.crawler.sitemap_crawler import SitemapCrawler
    from src.services.embeddings.embedding_service import EmbeddingService
    from src.services.ingestion.website_ingestor import WebsiteIngestor
    from src.services.vector_store.pgvector_store import PGVectorStore

    settings = get_settings()

    # ── startup banner ────────────────────────────────────
    LOGGER.info("=" * 60)
    LOGGER.info("ingest_worker starting")
    LOGGER.info("  bot_mode    : %s", settings.bot_mode)
    LOGGER.info("  website_url : %s", settings.website_url)
    LOGGER.info("  database    : %.40s...", settings.database_url)
    manual_paths = [p for p in settings.website_urls.split(",") if p.strip()]
    LOGGER.info("  manual_urls : %d paths configured", len(manual_paths))
    LOGGER.info("=" * 60)

    # ── build components (mirrors get_ingestion_pipeline) ─
    vector_store = PGVectorStore(WB_DATABASE_URL=settings.database_url)
    vector_store.initialize(vector_size=settings.embedding_dimensions)

    embedding_service = EmbeddingService(
        api_key=settings.nomic_api_key,
        model=settings.nomic_embedding_model,
    )

    crawler = SitemapCrawler(
        base_url=settings.website_url,
        user_agent=settings.user_agent,
        timeout=settings.request_timeout_seconds,
        manual_urls=settings.website_urls,
    )

    ingestor = WebsiteIngestor(
        crawler=crawler,
        embedding_service=embedding_service,
        vector_store=vector_store,
        chunk_size_words=settings.chunk_size_words,
        chunk_overlap_words=settings.chunk_overlap_words,
        enabled=settings.enable_website_ingest and bool(settings.website_url),
    )

    if not ingestor.is_enabled():
        LOGGER.warning(
            "Website ingestion is disabled. "
            "Check ENABLE_WEBSITE_INGEST and WEBSITE_URL in .env."
        )
        return

    # ── run ───────────────────────────────────────────────
    LOGGER.info("Starting ingestion run ...")
    t0 = time.time()

    result = ingestor.ingest()

    elapsed = time.time() - t0

    # ── final summary ─────────────────────────────────────
    LOGGER.info("=" * 60)
    LOGGER.info("Ingestion complete")
    LOGGER.info("  Pages crawled  : %d", result.processed_items)
    LOGGER.info("  Chunks stored  : %d", result.stored_chunks)
    LOGGER.info("  Elapsed        : %.1f s", elapsed)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
