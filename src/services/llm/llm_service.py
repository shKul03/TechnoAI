"""LLM response generation constrained to retrieved context using Groq."""

from __future__ import annotations

import json
import logging

import httpx

LOGGER = logging.getLogger(__name__)


def _fix_encoding(text: str) -> str:
    return (
        text.replace("â", "'")
            .replace("â", '"')
            .replace("â", '"')
            .replace("â", "—")
            .replace("â", "—")
            .replace("â", "—")
    )


_SYSTEM_PROMPT = """\
You are the AdaptHealth Assistant, a helpful and knowledgeable \
virtual assistant for AdaptHealth — a leading provider of home medical equipment (HME) \
and related services including sleep therapy, oxygen therapy, respiratory care, \
mobility equipment, diabetes supplies, wound care, and specialty care.

Your role is to help patients, caregivers, and healthcare providers find the \
information they need clearly and quickly.

IDENTITY RULES:
- You are the AdaptHealth Assistant
- If asked whether you are human or AI, say: "I'm an AI assistant here to help \
with AdaptHealth information."
- Never claim to be a doctor or provide medical diagnoses or treatment advice
- You can explain what equipment or services AdaptHealth provides, but always \
recommend consulting a healthcare provider for medical decisions

TONE AND STYLE:
- Warm, calm, and patient — many users are patients or caregivers dealing with \
health challenges
- Clear and plain language — avoid clinical jargon unless the user uses it first
- Concise — get to the point, then offer a next step
- Never sound sales-heavy, robotic, or dismissive

CONTACT RULE:
- If a user asks how to contact AdaptHealth, get in touch, or speak to someone, \
always provide: https://adapthealth.com/pages/contact-us
- For sleep therapy questions specifically, direct to: \
https://adapthealth.com/pages/contact-sleep-team

RESPONSE RULES:
- Answer based on AdaptHealth website content only
- If you don't have enough information, say so honestly and offer the contact page
- Always suggest a relevant next step — never leave the user at a dead end
- If a user expresses distress or mentions a medical emergency, acknowledge with \
empathy and direct them to their healthcare provider or 911 immediately
- Keep responses focused — do not pad answers with unnecessary caveats

CRITICAL FORMAT ENFORCEMENT:
- Do NOT start your response with "AdaptHealth", "The AdaptHealth", or "AdaptHealth's"
- Do NOT start with "Certainly", "Absolutely", "Of course", "Great question", \
or any hollow affirmation
- Do NOT use bullet points for every response — use prose when it reads more naturally
- Do NOT use markdown headers in conversational replies

OUT OF SCOPE:
- If asked about topics unrelated to AdaptHealth or home medical equipment, \
say: "I'm focused on AdaptHealth information. Is there something I can help you \
find on the AdaptHealth website?"
"""


class LLMService:
    """Wrap Groq generation with strict grounding instructions."""

    def __init__(
        self,
        groq_api_key: str,
        answer_model: str,
        rewrite_model: str,
        temperature: float,
        top_p: float,
        num_predict: int,
    ) -> None:
        self._api_key = groq_api_key
        self._answer_model = answer_model
        self._rewrite_model = rewrite_model
        self._temperature = temperature
        self._top_p = top_p
        self._num_predict = num_predict
        self._base_url = "https://api.groq.com/openai/v1"
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

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

        user_message = (
            f"{history_block}"
            f"Website content:\n{context}\n\n"
            "CRITICAL FORMAT ENFORCEMENT:\n"
            "- Plain text only. No markdown.\n"
            "- Never use **, *, #, or any markdown symbol.\n"
            "- Do not repeat yourself. Say each thing once.\n"
            "- If listing items, use ONLY hyphen bullets.\n"
            "- Do not write a prose sentence AND then repeat "
            "the same items as bullets. Choose one format.\n"
            "- Maximum 90 words total.\n"
            "- Do not start your answer with 'AdaptHealth' or 'At AdaptHealth'.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json={
                        "model": self._answer_model,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        "temperature": self._temperature,
                        "max_tokens": self._num_predict,
                        "stream": False,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                return _fix_encoding(data["choices"][0]["message"]["content"].strip())
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
            "AdaptHealth website chatbot.\n\n"
            "The user asked: " + question + "\n\n"
            "The website content used to answer was:\n"
            + context[:800] + "\n\n"
            "Generate 3 follow-up questions. Each must:\n"
            "1. Be answerable from the website content above "
            "— only ask about topics, services, or equipment "
            "explicitly named in the content\n"
            "2. Be broad — ask about a service area, equipment "
            "category, or patient topic, NOT about specific details, "
            "numbers, or how something was done\n"
            "3. Be about AdaptHealth home medical equipment and "
            "services, not about the user's personal situation\n"
            "4. Be under 10 words\n\n"
            "NEVER ask about: specific metrics, technical "
            "details of how something worked, names of systems "
            "used, or anything not in the content above.\n\n"
            "GOOD examples:\n"
            "[\"How do I reorder supplies?\","
            " \"What sleep equipment do you offer?\","
            " \"How do I contact AdaptHealth?\"]\n\n"
            "[\"Do you accept my insurance?\","
            " \"How do I set up my CPAP?\","
            " \"What mobility equipment is available?\"]\n\n"
            "BAD examples: 'What was the platform built with?', "
            "'How did the system handle data?', "
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
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json={
                        "model": self._rewrite_model,
                        "messages": [
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.4,
                        "max_tokens": 100,
                        "stream": False,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                raw = data["choices"][0]["message"]["content"].strip()
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
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json={
                        "model": self._rewrite_model,
                        "messages": [
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 50,
                        "stream": False,
                    },
                    timeout=30,
                )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"].strip()
            # Take only the first line and strip stray quotes/backticks
            result = result.split("\n")[0].strip().strip("\"'`")
            return result or None
        except Exception as exc:
            LOGGER.warning("[LLM] query rewrite failed: %s", exc)
            return None
