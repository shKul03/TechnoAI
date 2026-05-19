"""Graph node functions for Techno-AI LangGraph pipeline."""

from __future__ import annotations
import asyncio
import logging
import re

from src.graph.state import ChatState
from src.services.rag_service import (
    FALLBACK_RESPONSE,
    is_service_overview,
    service_slug_for_query,
    RAGService,
)
from src.dependencies import (
    get_llm_service,
    get_vector_store,
    get_embedding_service,
)
from src.config.settings import get_settings

LOGGER = logging.getLogger(__name__)

CONTACT_INTENT_RE = re.compile(
    r"\b(contact|email|phone|call|reach|get in touch|"
    r"office|location|address|speak to|talk to someone)\b",
    re.IGNORECASE,
)

CLOSURE_RE = re.compile(
    r"\b(thank|thanks|that.s all|that.s everything|bye|"
    r"goodbye|no more questions|i.m done|that.s it)\b",
    re.IGNORECASE,
)

CLOSURE_ANSWER = (
    "Thanks for chatting with us! If you ever have more "
    "questions about Technossus, feel free to ask anytime."
)


# ── Session ──────────────────────────────────────────────

def load_session(state: ChatState) -> dict:
    """
    On first turn: initialise empty chat_history.
    On returning turns: checkpointer already restored
    chat_history from Postgres — return it unchanged.
    """
    existing = state.get("chat_history")
    if existing is None:
        return {"chat_history": []}
    return {"chat_history": existing}


# ── Routing ──────────────────────────────────────────────

def intent_router(state: ChatState) -> dict:
    """
    Classify question intent.
    Returns one of: overview | contact | closure | general
    """
    question = state.get("question", "")
    if CLOSURE_RE.search(question):
        return {"intent": "closure"}
    if CONTACT_INTENT_RE.search(question):
        return {"intent": "contact"}
    if is_service_overview(question):
        return {"intent": "overview"}
    return {"intent": "general"}


async def rewrite_query(state: ChatState) -> dict:
    """
    Rewrite vague follow-up questions into standalone
    search queries using the dedicated rewrite model.
    Falls back to the raw question on failure.
    """
    llm = get_llm_service()
    question = state.get("question", "")
    history = state.get("chat_history", [])
    rewritten = await llm.rewrite_query(question, history)
    return {"retrieval_query": rewritten or question}


def slug_router(state: ChatState) -> dict:
    """
    Check if the retrieval query names a known page slug.
    Sets route to 'slug' or 'semantic'.
    """
    query = state.get("retrieval_query", "")
    slug = service_slug_for_query(query)
    if slug:
        return {"route": "slug", "service_slug": slug}
    return {"route": "semantic", "service_slug": None}


# ── Retrieval ─────────────────────────────────────────────

def fetch_overview(state: ChatState) -> dict:
    """
    Fetch the first chunk from each of the 6 service pages.
    Used for service overview questions.
    """
    vector_store = get_vector_store()
    service_slugs = [
        "ai-business-transformation",
        "cloud-product-modernization",
        "data-intelligence-analytics",
        "digital-experience-design",
        "product-engineering",
        "quality-engineering",
    ]
    chunks = []
    for slug in service_slugs:
        results = vector_store.get_by_url(slug)
        if results:
            chunks.append(results[0])
    context = RAGService._build_context(chunks)
    sources = [
        {"chunk_id": c.chunk_id, "score": c.score,
         "metadata": c.metadata}
        for c in chunks
    ]
    return {"chunks": chunks, "context": context,
            "sources": sources}


def fetch_by_slug(state: ChatState) -> dict:
    """
    Fetch all chunks for a known slug directly from the DB,
    bypassing the score threshold entirely.
    Applies leadership keyword filter for the about page.
    """
    vector_store = get_vector_store()
    slug = state.get("service_slug") or ""
    chunks = vector_store.get_by_url(slug)
    if slug == "about":
        leadership_chunks = [
            c for c in chunks
            if any(kw in c.content for kw in (
                "FOUNDER", "MANAGING PARTNER", "DIRECTOR",
                "PRESIDENT", "VICE PRESIDENT", "CEO",
            ))
        ]
        if leadership_chunks:
            chunks = leadership_chunks
    context = RAGService._build_context(chunks)
    sources = [
        {"chunk_id": c.chunk_id, "score": c.score,
         "metadata": c.metadata}
        for c in chunks
    ]
    return {"chunks": chunks, "context": context,
            "sources": sources}


def semantic_retrieve(state: ChatState) -> dict:
    """
    Embed the retrieval query and search pgvector.
    Returns raw results before score filtering.
    """
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()
    settings = get_settings()
    query = state.get("retrieval_query", "")
    embedding = embedding_service.embed_text(query)
    results = vector_store.search(
        embedding,
        top_k=settings.retrieval_top_k,
    )
    return {"chunks": results}


def score_gate(state: ChatState) -> dict:
    """
    Filter chunks below the score threshold.
    Sets route to 'llm' if chunks pass, 'fallback' if not.
    """
    settings = get_settings()
    chunks = state.get("chunks", [])
    relevant = [
        c for c in chunks
        if c.score >= settings.retrieval_min_score
    ]
    if not relevant:
        LOGGER.info("[graph] score_gate: no chunks passed threshold")
        return {
            "chunks": [],
            "context": "",
            "sources": [],
            "answer": FALLBACK_RESPONSE,
            "follow_ups": [],
            "route": "fallback",
        }
    context = RAGService._build_context(relevant)
    sources = [
        {"chunk_id": c.chunk_id, "score": c.score,
         "metadata": c.metadata}
        for c in relevant
    ]
    LOGGER.info(
        "[graph] score_gate: %d chunks passed", len(relevant)
    )
    return {
        "chunks": relevant,
        "context": context,
        "sources": sources,
        "route": "llm",
    }


# ── LLM ──────────────────────────────────────────────────

async def llm_answer(state: ChatState) -> dict:
    """
    Run answer_question and generate_follow_ups in parallel.
    Both calls use the same retrieved context.
    """
    llm = get_llm_service()
    question = state.get("question", "")
    context = state.get("context", "")
    history = state.get("chat_history", [])

    answer_text, follow_ups = await asyncio.gather(
        llm.answer_question(question, context, history),
        llm.generate_follow_ups(question, context, history),
    )
    return {
        "answer": answer_text or FALLBACK_RESPONSE,
        "follow_ups": follow_ups,
    }


def contact_response(state: ChatState) -> dict:
    """
    Return the hardcoded contact details.
    Contact info is sourced from the website footer,
    not from ingested chunks, to avoid placeholder numbers.
    """
    answer = (
        "You can reach us at contact@technossus.com or call "
        "+1 (949) 769-3500. You can also visit our contact "
        "page at https://technossus.com/contact to fill out "
        "a form and our team will get back to you."
    )
    return {"answer": answer, "follow_ups": [], "sources": []}


def closure_response(state: ChatState) -> dict:
    """Return a friendly closing message with no follow-ups."""
    return {
        "answer": CLOSURE_ANSWER,
        "follow_ups": [],
        "sources": [],
    }


# ── Session save ─────────────────────────────────────────

def save_session(state: ChatState) -> dict:
    """
    Append the current turn to chat_history in graph
    state. LangGraph checkpointer persists this to
    Postgres automatically.
    """
    question = state.get("question", "")
    answer = state.get("answer", "")
    history = list(state.get("chat_history") or [])

    if answer and answer != FALLBACK_RESPONSE:
        history.append(f"User: {question}")
        history.append(f"Assistant: {answer}")
        # Keep last 6 entries (3 turns)
        history = history[-6:]

    return {"chat_history": history}
