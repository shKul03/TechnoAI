"""LLM response generation constrained to retrieved context using Ollama."""

from __future__ import annotations

import json
import logging

import httpx

LOGGER = logging.getLogger(__name__)


def _fix_encoding(text: str) -> str:
    return (
        text.replace("â", "'")
            .replace("â", '"')
            .replace("â", '"')
    )


_SYSTEM_PROMPT = """\
You are Techno-AI, the intelligent assistant built into the Technossus
website. Speak as Technossus — use "we", "our", "at Technossus".
Never say "they" or "the company".

GROUNDING RULE — this is your most important instruction:
Answer ONLY using the website content provided below. Every claim,
statistic, name, percentage, and detail in your response MUST appear
verbatim or be directly inferable from that content. If the content
does not clearly support the answer, you MUST respond with exactly
this message and nothing else:
I can help with questions based on Technossus website content. You can ask about our services, industries, case studies, leadership, or AI capabilities.

CONTACT RULE:
This rule ONLY applies when the user's question explicitly asks for contact information, a phone number, an email address, office locations, or how to get in touch.
For ALL other questions, ignore this rule entirely and do not mention contact details.
When the rule applies, respond with exactly this and nothing else:
You can reach us at contact@technossus.com or call +1 (949) 769-3500. You can also visit our contact page at https://technossus.com/contact to fill out a form and our team will get back to you.

Never invent, estimate, or extrapolate statistics, percentages,
dates, names, or outcomes. If a number is not in the content,
it does not exist.

Banned phrases — never use these:
- according to
- based on
- provided context
- website content says
- it is worth noting
- not exhaustive
- additionally
- furthermore

Format rules:
- Answer in 60-90 words. Be concise.
- Plain text only. No markdown of any kind.
- Never use *, **, #, or any markdown symbol.
- Never number a list. Use only hyphen bullets if a list is needed.
- For service overview questions: maximum 6 hyphen bullets.
- For follow-up questions: answer only the one referenced item,
  60-90 words, plain text.
- Do not end with a question. Never append a follow-up question to your answer. Follow-up suggestions are handled separately.

If you are ever unsure whether the content supports your answer,
default to the fallback message above. An honest fallback is
better than a plausible-sounding fabrication.\
"""


class LLMService:
    """Wrap Ollama generation with strict grounding instructions."""

    def __init__(
        self,
        base_url: str,
        model: str,
        rewrite_model: str,
        temperature: float,
        top_p: float,
        num_predict: int,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._rewrite_model = rewrite_model
        self._temperature = temperature
        self._top_p = top_p
        self._num_predict = num_predict

    async def answer_question(
        self,
        question: str,
        context: str,
        chat_history: list[str] | None = None,
    ) -> str:
        """Generate an answer constrained to retrieved context."""

        history_block = ""
        if chat_history:
            history_block = "Conversation history:\n" + "\n".join(chat_history) + "\n\n"

        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"{history_block}"
            f"Website content:\n{context}\n\n"
            "CRITICAL GROUNDING INSTRUCTION:\n"
            "Your answer must be built EXCLUSIVELY from the Website "
            "content section above. Every service name, statistic, "
            "percentage, and detail must appear word-for-word in that "
            "content. If a service name in your answer does not appear "
            "in the Website content above, you are hallucinating — "
            "stop and use the fallback message instead.\n"
            "Fallback message: I can help with questions based on "
            "Technossus website content. You can ask about our services, "
            "industries, case studies, leadership, or AI capabilities.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self._temperature,
                            "top_p": self._top_p,
                            "num_predict": self._num_predict,
                        },
                    },
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                return _fix_encoding(data.get("response", "").strip())
        except Exception as exc:
            LOGGER.error("[LLM] answer_question failed: %s", exc)
            return "I'm having trouble connecting right now. Please try again in a moment."

    async def generate_follow_ups(
        self,
        question: str,
        context: str,
        chat_history: list[str] | None = None,
    ) -> list[str]:
        """Generate 2-3 contextually relevant follow-up suggestions."""
        prompt = (
            "You generate follow-up suggestions for the "
            "Technossus website chatbot.\n\n"
            "The user asked: " + question + "\n\n"
            "The website content used to answer was:\n"
            + context[:800] + "\n\n"
            "Generate 3 follow-up questions. Each must:\n"
            "1. Be answerable from the website content above "
            "— only ask about topics, services, or companies "
            "explicitly named in the content\n"
            "2. Be broad — ask about a service area, industry, "
            "or company topic, NOT about specific details, "
            "numbers, or how something was done\n"
            "3. Be about Technossus offerings, not about the "
            "user's situation\n"
            "4. Be under 10 words\n\n"
            "NEVER ask about: specific metrics, technical "
            "details of how something worked, names of systems "
            "used, or anything not in the content above.\n\n"
            "GOOD examples: 'What industries do you serve?', "
            "'Tell me about healthcare services.', "
            "'Do you work with financial services?'\n\n"
            "BAD examples: 'What was the platform built with?', "
            "'How did the CRM handle data?', "
            "'What were the specific results?'\n\n"
            "IMPORTANT: Do not suggest the question that was just "
            "asked. The user already asked: " + question + "\n\n"
            "Return ONLY a JSON array of 3 strings. "
            "No explanation, no markdown.\n"
            "JSON array:"
        )
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._rewrite_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.4,
                            "num_predict": 80,
                        },
                    },
                    timeout=15,
                )
                response.raise_for_status()
                raw = response.json().get("response", "").strip()
                raw = raw.strip("```json").strip("```").strip()
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [_fix_encoding(str(s)) for s in parsed[:3]]
                return []
        except Exception as exc:
            LOGGER.warning("[LLM] follow_ups generation failed: %s", exc)
            return []

    async def rewrite_query(self, question: str, chat_history: list[str]) -> str | None:
        """Rewrite a vague follow-up into a standalone search query.

        Returns a short query string, or None if the call fails or returns empty.
        Uses temperature=0 and a small token budget — this is a lookup, not generation.
        """
        FOLLOW_UP_SIGNALS = (
            "first", "second", "third", "fourth", "fifth", "sixth",
            "that", "this", "it", "the one", "previous", "last",
            "more", "tell me more", "explain", "elaborate", "expand",
            "which", "what about", "how about",
        )
        question_lower = question.lower()
        if not any(signal in question_lower for signal in FOLLOW_UP_SIGNALS):
            return None

        history_text = "\n".join(chat_history)
        prompt = (
            "Given the conversation history and the latest user message, "
            "rewrite the latest user message into a short, specific, "
            "standalone search query. Resolve any pronouns or references "
            "like 'the first one', 'that service', 'it' using the "
            "conversation above.\n\n"
            "Return only the search query.\n"
            "No explanation.\n"
            "No markdown.\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"Latest message:\n{question}\n\n"
            "Search query:"
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._rewrite_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 50,
                        },
                    },
                    timeout=30,
                )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            # Take only the first line and strip stray quotes/backticks
            result = result.split("\n")[0].strip().strip("\"'`")
            return result or None
        except Exception as exc:
            LOGGER.warning("[LLM] query rewrite failed: %s", exc)
            return None
