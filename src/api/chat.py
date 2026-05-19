"""Chat API routes."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from src.graph.state import ChatState
from src.limiter import limiter
from src.services import chat_memory
from src.services.rag_service import FALLBACK_RESPONSE

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None)


class ChatResponse(BaseModel):
    """Outgoing chat response payload."""

    answer: str
    sources: list[dict]
    session_id: str
    follow_ups: list[str] = Field(default_factory=list)


@router.post("", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """Answer a question using the RAG pipeline, with optional session memory."""

    session_id = payload.session_id or str(uuid.uuid4())

    initial_state: ChatState = {
        "question": payload.question,
        "session_id": session_id,
        "intent": "general",
        "route": "semantic",
        "retrieval_query": "",
        "service_slug": None,
        "chunks": [],
        "context": "",
        "sources": [],
        "answer": "",
        "follow_ups": [],
    }

    config = RunnableConfig(
        configurable={"thread_id": session_id}
    )

    try:
        from src.graph import graph as graph_module
        if graph_module.chat_graph is None:
            raise HTTPException(
                status_code=503,
                detail="Service starting up. Please retry."
            )
        result = await graph_module.chat_graph.ainvoke(
            initial_state,
            config=config,
        )
        return ChatResponse(
            answer=result.get("answer") or FALLBACK_RESPONSE,
            sources=result.get("sources", []),
            session_id=session_id,
            follow_ups=result.get("follow_ups", []),
        )
    except Exception as exc:
        LOGGER.exception("Chat graph invocation failed")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again.",
        ) from exc


@router.delete("/session/{session_id}", tags=["chat"])
def clear_session(session_id: str) -> dict[str, str]:
    """Clear chat memory for a given session."""

    found = chat_memory.clear_session(session_id)
    if not found:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "cleared", "session_id": session_id}
