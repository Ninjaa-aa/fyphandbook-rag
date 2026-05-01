"""Groq LLM client wrapper with retry on rate limits / transient errors."""

from __future__ import annotations

import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

from .config import LLM_MAX_TOKENS, LLM_MODEL, LLM_RETRIES, LLM_TEMPERATURE
from .logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_client() -> Groq:
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=key)


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """Call Groq chat completions with bounded retries."""
    client = get_client()
    last_err: Exception | None = None
    backoff = 1.0

    for attempt in range(LLM_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if attempt == LLM_RETRIES:
                break
            log.warning("Groq call failed (attempt %d/%d): %s", attempt + 1, LLM_RETRIES + 1, e)
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(f"Groq generation failed after {LLM_RETRIES + 1} attempts: {last_err}")
