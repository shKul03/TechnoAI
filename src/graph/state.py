"""LangGraph state schema for Techno-AI chat graph."""

from __future__ import annotations
from typing import TypedDict


class ChatState(TypedDict, total=False):
    """
    Shared state passed between all graph nodes.
    All fields are optional (total=False) so nodes can
    return partial updates — LangGraph merges them.
    """

    # Input — set at graph entry point
    question: str
    session_id: str

    # Session — populated by load_session node
    chat_history: list[str]

    # Routing — set by intent_router, slug_router
    intent: str        # overview | contact | closure | general
    route: str         # slug | semantic | llm | fallback
    retrieval_query: str
    service_slug: str | None

    # Retrieval — populated by fetch nodes
    chunks: list       # list[SearchResult]
    context: str
    sources: list[dict]

    # Output — populated by LLM nodes
    answer: str
    follow_ups: list[str]
