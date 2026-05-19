"""LangGraph graph definition for Techno-AI."""

from __future__ import annotations
import logging
from langgraph.graph import StateGraph, END
from src.graph.state import ChatState
from src.graph.nodes import (
    load_session,
    intent_router,
    rewrite_query,
    slug_router,
    fetch_overview,
    fetch_by_slug,
    semantic_retrieve,
    score_gate,
    llm_answer,
    contact_response,
    closure_response,
    save_session,
)

LOGGER = logging.getLogger(__name__)

# Module-level reference — populated at app startup
# by initialise_graph(). None until then.
chat_graph = None
_connection_pool = None


def _route_intent(state: ChatState) -> str:
    return state.get("intent", "general")


def _route_slug(state: ChatState) -> str:
    return state.get("route", "semantic")


def _route_score(state: ChatState) -> str:
    return state.get("route", "llm")


def _build_graph_definition() -> StateGraph:
    """Build the graph topology (nodes + edges).
    Does not compile — caller provides checkpointer."""
    graph = StateGraph(ChatState)

    graph.add_node("load_session",      load_session)
    graph.add_node("intent_router",     intent_router)
    graph.add_node("rewrite_query",     rewrite_query)
    graph.add_node("slug_router",       slug_router)
    graph.add_node("fetch_overview",    fetch_overview)
    graph.add_node("fetch_by_slug",     fetch_by_slug)
    graph.add_node("semantic_retrieve", semantic_retrieve)
    graph.add_node("score_gate",        score_gate)
    graph.add_node("llm_answer",        llm_answer)
    graph.add_node("contact_response",  contact_response)
    graph.add_node("closure_response",  closure_response)
    graph.add_node("save_session",      save_session)

    graph.set_entry_point("load_session")
    graph.add_edge("load_session", "intent_router")

    graph.add_conditional_edges(
        "intent_router",
        _route_intent,
        {
            "overview": "fetch_overview",
            "contact":  "contact_response",
            "closure":  "closure_response",
            "general":  "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query", "slug_router")

    graph.add_conditional_edges(
        "slug_router",
        _route_slug,
        {
            "slug":     "fetch_by_slug",
            "semantic": "semantic_retrieve",
        },
    )

    graph.add_edge("fetch_overview",    "llm_answer")
    graph.add_edge("fetch_by_slug",     "llm_answer")
    graph.add_edge("semantic_retrieve", "score_gate")

    graph.add_conditional_edges(
        "score_gate",
        _route_score,
        {
            "llm":      "llm_answer",
            "fallback": "save_session",
        },
    )

    graph.add_edge("llm_answer",       "save_session")
    graph.add_edge("contact_response", "save_session")
    graph.add_edge("closure_response", "save_session")
    graph.add_edge("save_session",     END)

    return graph


async def initialise_graph(database_url: str) -> None:
    """
    Compile the graph with AsyncPostgresSaver and store
    it in the module-level chat_graph reference.
    Called once at FastAPI startup.
    """
    global chat_graph, _connection_pool

    import psycopg
    from typing import Any
    from psycopg.rows import dict_row
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    # Run setup() on a direct connection with autocommit=True
    # because CREATE INDEX CONCURRENTLY cannot run inside
    # a transaction block (which the pool creates by default)
    async with await psycopg.AsyncConnection.connect(
        database_url,
        autocommit=True,
        row_factory=dict_row,
    ) as conn:
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()

    # Now create the pool for runtime use
    pool: AsyncConnectionPool[psycopg.AsyncConnection[dict[str, Any]]] = AsyncConnectionPool(
        conninfo=database_url,
        max_size=10,
        min_size=1,
        open=False,
        kwargs={
            "row_factory": dict_row,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )
    await pool.open()
    _connection_pool = pool  # store for shutdown

    # Compile graph with pool-backed checkpointer
    runtime_checkpointer = AsyncPostgresSaver(pool)
    graph_def = _build_graph_definition()
    chat_graph = graph_def.compile(checkpointer=runtime_checkpointer)

    LOGGER.info("[graph] AsyncPostgresSaver initialised")


async def shutdown_graph() -> None:
    """Close the connection pool gracefully at shutdown."""
    global _connection_pool
    if _connection_pool is not None:
        await _connection_pool.close()
        LOGGER.info("[graph] connection pool closed")
