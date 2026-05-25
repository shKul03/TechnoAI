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
    "I can help with questions about AdaptHealth's home medical equipment and services. "
    "You can ask about sleep therapy, oxygen therapy, respiratory care, mobility equipment, "
    "diabetes supplies, wound care, specialty care, or how to get started as a patient."
)

# Canonical service list — deterministic, so the answer is always consistent
# and follow-up ordinal resolution works reliably.
_CANONICAL_SERVICES = [
    "sleep therapy",
    "oxygen therapy",
    "respiratory care",
    "mobility equipment",
    "diabetes supplies",
    "wound care",
    "specialty care",
    "incontinence",
    "ostomy",
    "urology",
]

# Matches common ways a user asks for AdaptHealth's products/services overview
_SERVICE_OVERVIEW_RE = re.compile(
    r"""
    (?:what|which|list|tell\s+me\s+about)\s+
    (?:are\s+(?:your|the)|do\s+you\s+offer|does\s+adapt.{0,10}health\s+offer)?
    \s*(?:services?|products?|equipment|solutions?|offerings?)
    |what\s+do\s+you\s+(?:offer|provide|do|supply)
    |what\s+can\s+adapt.{0,10}health\s+(?:help|do|offer|provide)
    |(?:home\s+medical|dme|hme)\s+(?:equipment|supplies?)
    |tell\s+me\s+about\s+adapt.{0,10}health
    |what\s+is\s+adapt.{0,10}health
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_service_overview(question: str) -> bool:
    return bool(_SERVICE_OVERVIEW_RE.search(question))


# Maps user query substrings → URL slug fragment (AdaptHealth pages).
# Longer/more-specific keys are listed first so they match before shorter
# substrings (e.g. "sleep apnea" before "sleep").
# Used to boost retrieval precision when a rewrite query names a specific page.
_SERVICE_SLUG_MAP: dict[str, str] = {
    # Sleep / CPAP / PAP
    "sleep apnea":            "pages/sleep-apnea",
    "sleep therapy":          "pages/sleep-apnea",
    "cpap":                   "pages/sleep-apnea",
    "bipap":                  "pages/sleep-apnea",
    "apap":                   "pages/sleep-apnea",
    "pap therapy":            "pages/sleep-apnea",
    "pap reorder":            "pages/pap-reorder",
    "cpap supplies":          "pages/pap-reorder",
    "pap cleaning":           "pages/pap-cleaning-schedule",
    "sleep products":         "pages/sleep-products-support",
    "sleep support":          "pages/sleep-products-support",
    "sleep health":           "pages/sleep-health",
    "sleep success":          "pages/sleep-success",
    # Oxygen
    "oxygen therapy":         "pages/oxygen-therapy",
    "home oxygen":            "pages/oxygen-therapy",
    "supplemental oxygen":    "pages/oxygen-therapy",
    "oxygen":                 "pages/oxygen-therapy",
    # Respiratory
    "respiratory care":       "pages/respiratory-care",
    "ventilator":             "pages/respiratory-care",
    "nebulizer":              "pages/respiratory-care",
    "respiratory":            "pages/respiratory-care",
    # Mobility
    "mobility":               "pages/mobility-home-equipment",
    "wheelchair":             "pages/mobility-home-equipment",
    "scooter":                "pages/mobility-home-equipment",
    "walker":                 "pages/mobility-home-equipment",
    "crutches":               "pages/mobility-home-equipment",
    "hospital bed":           "pages/mobility-home-equipment",
    "home equipment":         "pages/mobility-home-equipment",
    # Diabetes
    "diabetes supplies":      "pages/diabetes-express-reorder",
    "glucose":                "pages/diabetes-express-reorder",
    "cgm":                    "pages/diabetes-express-reorder",
    "diabetes":               "pages/diabetes-express-reorder",
    # Specialty / Other products
    "wound care":             "pages/wound-care",
    "wound":                  "pages/wound-care",
    "ostomy":                 "pages/ostomy",
    "urological":             "pages/urology",
    "urology":                "pages/urology",
    "incontinence":           "pages/incontinence",
    "specialty care":         "pages/specialty-care",
    "rehabilitation":         "pages/adaptrehab",
    "rehab":                  "pages/adaptrehab",
    "wellness":               "pages/wellness-at-home",
    # Ordering / Patient tools
    "express reorder":        "pages/express-reorder",
    "order supplies":         "pages/express-reorder",
    "reorder":                "pages/express-reorder",
    "new patient":            "pages/new-patient-packet",
    "getting started":        "pages/new-patient-packet",
    "patient welcome":        "pages/patient-welcome-guide",
    "app tutorial":           "pages/myapp-tutorials",
    "my app":                 "pages/myapp",
    "mobile app":             "pages/myapp",
    "myapp":                  "pages/myapp",
    "electronic prescription": "pages/eprescribe",
    "e-prescribe":            "pages/eprescribe",
    "eprescribe":             "pages/eprescribe",
    "pay bill":               "pages/pay-your-bill",
    "billing":                "pages/pay-your-bill",
    "insurance card":         "pages/insurance-card",
    # Insurance / Payers
    "insurance companies":    "pages/insurance-companies",
    "accepted insurance":     "pages/insurance-companies",
    "insurance":              "pages/insurance-companies",
    "humana":                 "pages/humana",
    "kaiser":                 "pages/kaiser",
    # Partners / Equipment brands
    "philips recall":         "pages/philipsrecall",
    "respironics recall":     "pages/philipsrecall",
    "philips":                "pages/philips-respironics",
    "resmed":                 "pages/resmed",
    "fisher paykel":          "pages/fisher-paykel",
    "react health":           "pages/react-health",
    "solara":                 "pages/solara",
    # Contact / Locations
    "contact us":             "pages/contact-us",
    "sleep team":             "pages/contact-sleep-team",
    "find a location":        "pages/locations",
    "locations":              "pages/locations",
    "minnesota":              "pages/minnesota",
    "new england":            "pages/newengland",
    "contact":                "pages/contact-us",
    # About / Corporate
    "about adapthealth":      "pages/about-us",
    "mission":                "pages/mission-vision-values",
    "vision":                 "pages/mission-vision-values",
    "values":                 "pages/mission-vision-values",
    "accreditation":          "pages/accreditation",
    "compliance":             "pages/corporate-compliance",
    "leadership":             "pages/leadership-team",
    "executives":             "pages/leadership-team",
    "team":                   "pages/leadership-team",
    "careers":                "pages/careers",
    "jobs":                   "pages/careers",
    "sustainability":         "pages/esg-overview",
    "esg":                    "pages/esg-overview",
    "about":                  "pages/about-us",
    # Investor relations
    "investor relations":     "pages/investor-relations",
    "investors":              "pages/investor-relations",
    "investor":               "pages/investor-relations",
    "stock":                  "pages/stock-information",
    "financials":             "pages/financials",
    "governance":             "pages/governance",
}


def _service_slug_for_query(query: str) -> str | None:
    """Return the URL slug if the query clearly names one specific page."""
    lower = query.lower()
    for name, slug in _SERVICE_SLUG_MAP.items():
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
