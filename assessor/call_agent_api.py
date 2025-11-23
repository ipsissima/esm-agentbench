"""Resilient OpenAI chat wrapper with deterministic fallback.

This helper centralizes API handling so assessor logic stays clean. It prefers
the modern ``openai`` client when available, falls back to the legacy module,
and always returns deterministic text if credentials are missing or the API
errors. Errors are logged verbosely when ``DEBUG_CALL_AGENT`` is set.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None


def _build_client():
    if openai is None:
        return None
    try:  # pragma: no cover - optional dependency
        if hasattr(openai, "OpenAI"):
            return openai.OpenAI()
    except Exception:
        return None
    return None


def call_agent_api(
    prompt: str,
    deterministic_fallback: Optional[Callable[[str], str]] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """Call the OpenAI chat API if credentials exist, else fallback.

    Parameters
    ----------
    prompt: str
        User message to send to the chat completion API.
    deterministic_fallback: callable, optional
        Function used when API is unavailable or fails. Should accept a prompt
        and return a string.
    model: str
        Preferred model name when using the modern SDK.
    """

    api_key = os.environ.get("OPENAI_API_KEY")
    debug = os.environ.get("DEBUG_CALL_AGENT", "0") == "1"

    if api_key and openai is not None:
        try:  # pragma: no cover - external network
            client = _build_client()
            if client is not None:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                content = resp.choices[0].message.content or ""
            else:
                completion = openai.ChatCompletion.create(  # type: ignore[attr-defined]
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                content = completion["choices"][0]["message"]["content"]
            return content
        except Exception as exc:  # pragma: no cover - defensive
            if debug:
                logger.exception("OpenAI call failed: %s", exc)
            else:
                logger.warning("OpenAI call failed; using deterministic fallback.")

    if deterministic_fallback is not None:
        return deterministic_fallback(prompt)
    return ""


__all__ = ["call_agent_api"]
