"""Dependency wiring for the application."""

from __future__ import annotations

from src.config.settings import get_settings
from src.services.crawler.sitemap_crawler import SitemapCrawler
from src.services.embeddings.embedding_service import EmbeddingService
from src.services.ingestion.file_ingestor import FileIngestor
from src.services.ingestion.pipeline import IngestionPipeline
from src.services.ingestion.website_ingestor import WebsiteIngestor
from src.services import chat_memory as _chat_memory_module
from src.services.llm.llm_service import LLMService
from src.services.rag_service import RAGService
from src.services.vector_store.base import VectorStore
from src.services.vector_store.pgvector_store import PGVectorStore


def get_vector_store() -> VectorStore:
    """Build the configured vector store.

    Reads VECTOR_STORE_TYPE from settings.  Currently only "pgvector" is
    supported; the ChromaDB path has been removed.
    """

    settings = get_settings()
    store_type = settings.vector_store_type.lower()
    if store_type == "pgvector":
        return PGVectorStore(database_url=settings.database_url)
    raise ValueError(
        f"Unsupported VECTOR_STORE_TYPE '{store_type}'. "
        "Set VECTOR_STORE_TYPE=pgvector in your .env file."
    )


def get_embedding_service() -> EmbeddingService:
    """Build the embedding service."""

    settings = get_settings()
    return EmbeddingService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


def get_llm_service() -> LLMService:
    """Build the LLM service."""

    settings = get_settings()
    return LLMService(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        rewrite_model=settings.ollama_rewrite_model,
        temperature=settings.ollama_temperature,
        top_p=settings.ollama_top_p,
        num_predict=settings.ollama_num_predict,
    )


def get_rag_service() -> RAGService:
    """Build the RAG service."""

    settings = get_settings()
    return RAGService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        llm_service=get_llm_service(),
        retrieval_top_k=settings.retrieval_top_k,
        retrieval_min_score=settings.retrieval_min_score,
    )


def get_chat_memory():
    """Return the chat memory module for use in graph nodes."""
    return _chat_memory_module


def get_ingestion_pipeline() -> IngestionPipeline:
    """Build the ingestion pipeline with enabled sources."""

    settings = get_settings()
    vector_store = get_vector_store()
    embedding_service = get_embedding_service()
    crawler = SitemapCrawler(
        base_url=settings.website_url,
        user_agent=settings.user_agent,
        timeout=settings.request_timeout_seconds,
        manual_urls=settings.website_urls,
    )
    sources = [
        WebsiteIngestor(
            crawler=crawler,
            embedding_service=embedding_service,
            vector_store=vector_store,
            chunk_size_words=settings.chunk_size_words,
            chunk_overlap_words=settings.chunk_overlap_words,
            enabled=settings.enable_website_ingest and bool(settings.website_url),
        ),
        FileIngestor(
            embedding_service=embedding_service,
            vector_store=vector_store,
            pdf_directory=settings.pdf_directory,
            chunk_size_words=settings.chunk_size_words,
            chunk_overlap_words=settings.chunk_overlap_words,
            enabled=settings.enable_file_ingest,
        ),
    ]
    return IngestionPipeline(sources=sources)
