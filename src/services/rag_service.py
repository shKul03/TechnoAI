"""RAG query orchestration."""

from __future__ import annotations

import asyncio
import logging
import re

from src.services.embeddings.embedding_service import EmbeddingService
from src.services.llm.llm_service import LLMService
from src.services.vector_store.base import SearchResult, VectorStore

LOGGER = logging.getLogger(__name__)

FALLBACK_RESPONSE = (
    "I can help with questions based on Technossus website content. "
    "You can ask about our services, industries, case studies, leadership, or AI capabilities."
)

# Canonical service list — deterministic, so the answer is always consistent
# and follow-up ordinal resolution works reliably.
_CANONICAL_SERVICES = [
    "AI Business Transformation",
    "Cloud & Product Modernization",
    "Data Intelligence & Analytics",
    "Digital Experience Design",
    "Product Engineering",
    "Quality Engineering",
]

# Matches common ways a user asks for the services list
_SERVICE_OVERVIEW_RE = re.compile(
    r"\b("
    r"what services|which services|list (your |the )?services|"
    r"services (do you|does technossus|you) offer|"
    r"what do you offer|what (can|does) technossus (do|offer|help)|"
    r"(your |the )?services (you |technossus )?(provide|have)|"
    r"tell me (about )?(your |the )?services|"
    r"overview of (your |the )?services"
    r")\b",
    re.IGNORECASE,
)


def _is_service_overview(question: str) -> bool:
    return bool(_SERVICE_OVERVIEW_RE.search(question))


# Maps canonical service name substrings → URL slug fragment.
# Used to boost retrieval precision when a rewrite query names a specific service.
_SERVICE_SLUG_MAP: list[tuple[str, str]] = [
    ("ai business transformation", "ai-business-transformation"),
    ("cloud & product modernization", "cloud-product-modernization"),
    ("cloud and product modernization", "cloud-product-modernization"),
    ("data intelligence & analytics", "data-intelligence-analytics"),
    ("data intelligence and analytics", "data-intelligence-analytics"),
    ("digital experience design", "digital-experience-design"),
    ("product engineering", "product-engineering"),
    ("quality engineering", "quality-engineering"),
    ("case study", "case-studies"),
    ("case studies", "case-studies"),
    ("our work", "case-studies"),
    ("testimonial", "about"),
    ("clients say", "about"),
    ("what clients", "about"),
    ("contact", "contact"),
    ("reach out", "contact"),
    ("get in touch", "contact"),
    ("phone", "contact"),
    ("email", "contact"),
    ("office", "contact"),
    ("location", "contact"),
    ("leadership", "about"),
    ("leaders", "about"),
    ("executive", "about"),
    ("team members", "about"),
    ("who is", "about"),
]


def _service_slug_for_query(query: str) -> str | None:
    """Return the URL slug if the query clearly names one specific service."""
    lower = query.lower()
    for name, slug in _SERVICE_SLUG_MAP:
        if name in lower:
            return slug
    return None


def is_service_overview(question: str) -> bool:
    return _is_service_overview(question)


def service_slug_for_query(query: str) -> str | None:
    return _service_slug_for_query(query)


class RAGService:
    """Retrieve relevant chunks and generate grounded responses."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
        retrieval_top_k: int,
        retrieval_min_score: float,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._llm_service = llm_service
        self._retrieval_top_k = retrieval_top_k
        self._retrieval_min_score = retrieval_min_score

    async def answer(self, question: str, chat_history: list[str] | None = None) -> dict:
        """Run the RAG pipeline and return the assistant answer."""

        history = chat_history or []

        if _is_service_overview(question):
            LOGGER.info("[RAG] service overview — fetching all service chunks")
            service_slugs = [
                "ai-business-transformation",
                "cloud-product-modernization",
                "data-intelligence-analytics",
                "digital-experience-design",
                "product-engineering",
                "quality-engineering",
            ]
            overview_chunks = []
            for slug in service_slugs:
                chunks = self._vector_store.get_by_url(slug)
                if chunks:
                    overview_chunks.append(chunks[0])

            if overview_chunks:
                context = self._build_context(overview_chunks)
                answer_text, follow_ups = await asyncio.gather(
                    self._llm_service.answer_question(
                        question,
                        context,
                        history,
                    ),
                    self._llm_service.generate_follow_ups(
                        question,
                        context,
                        history,
                    ),
                )
                return {
                    "answer": answer_text or FALLBACK_RESPONSE,
                    "sources": [
                        {
                            "chunk_id": c.chunk_id,
                            "score": c.score,
                            "metadata": c.metadata,
                        }
                        for c in overview_chunks
                    ],
                    "follow_ups": follow_ups,
                }

        LOGGER.info("[RAG] user message: %r", question)

        # SLM rewrite — always fires before retrieval.
        # The SLM resolves ordinals ("3rd one"), vague refs ("yes please",
        # "tell me more"), and explicit questions alike. Falls back to the
        # raw user message if the call fails or returns empty output.
        retrieval_query = (
            await self._llm_service.rewrite_query(question, history) or question
        )
        LOGGER.info("[RAG] retrieval query: %r", retrieval_query)

        service_slug = _service_slug_for_query(retrieval_query)
        if service_slug:
            slug_chunks = self._vector_store.get_by_url(service_slug)
            if slug_chunks:
                if service_slug == "about":
                    leadership_chunks = [
                        c for c in slug_chunks
                        if any(kw in c.content for kw in (
                            "FOUNDER", "MANAGING PARTNER", "DIRECTOR",
                            "PRESIDENT", "VICE PRESIDENT", "CEO",
                        ))
                    ]
                    if leadership_chunks:
                        slug_chunks = leadership_chunks
                LOGGER.info(
                    "[RAG] direct slug fetch: slug=%s returned %d chunks",
                    service_slug, len(slug_chunks),
                )
                slug_context = self._build_context(slug_chunks)
                answer_text, follow_ups = await asyncio.gather(
                    self._llm_service.answer_question(
                        question,
                        slug_context,
                        history,
                    ),
                    self._llm_service.generate_follow_ups(
                        question,
                        slug_context,
                        history,
                    ),
                )
                answer = answer_text or FALLBACK_RESPONSE
                return {
                    "answer": answer,
                    "sources": [
                        {
                            "chunk_id": c.chunk_id,
                            "score": c.score,
                            "metadata": c.metadata,
                        }
                        for c in slug_chunks
                    ],
                    "follow_ups": follow_ups,
                }

        query_embedding = self._embedding_service.embed_text(retrieval_query)
        search_results = self._vector_store.search(
            embedding=query_embedding,
            top_k=self._retrieval_top_k,
        )

        relevant_results = [
            r for r in search_results if r.score >= self._retrieval_min_score
        ]

        if not relevant_results:
            LOGGER.info("[RAG] no results above min_score threshold")
            return {"answer": FALLBACK_RESPONSE, "sources": [], "follow_ups": []}

        for r in relevant_results:
            src = r.metadata.get("url") or r.metadata.get("file_name", "?")
            LOGGER.info("[RAG] retrieved: score=%.4f  source=%s", r.score, src)

        context = self._build_context(relevant_results)
        # Original question goes to the answer prompt; the SLM rewrite was
        # retrieval-only. Chat history gives the LLM enough context to understand
        # what the user meant by vague follow-ups.
        answer_text, follow_ups = await asyncio.gather(
            self._llm_service.answer_question(
                question=question,
                context=context,
                chat_history=history,
            ),
            self._llm_service.generate_follow_ups(
                question,
                context,
                history,
            ),
        )
        answer = answer_text or FALLBACK_RESPONSE
        return {
            "answer": answer,
            "sources": [
                {
                    "chunk_id": result.chunk_id,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in relevant_results
            ],
            "follow_ups": follow_ups,
        }

    @staticmethod
    def _build_context(results: list[SearchResult]) -> str:
        context_parts = []
        for result in results:
            source_ref = result.metadata.get("url") or result.metadata.get(
                "file_name",
                "",
            )
            context_parts.append(f"Source: {source_ref}\nContent: {result.content}")
        return "\n\n".join(context_parts)
