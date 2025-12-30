"""Perturbation-based robustness certification for embeddings.

This module computes the local Lipschitz constant of the embedding function
under semantic invariance by generating semantically equivalent perturbations
of input text and measuring embedding divergence.

Mathematical Basis:
The Lipschitz margin is defined as:
    lipschitz_margin = max_i { ||embed(p_i) - embed(step_i)|| }
where p_i are semantic perturbations of step_i.

This margin is then incorporated into the theoretical bound:
    theoretical_bound = C_res * residual + C_tail * tail_energy +
                       C_sem * semantic_divergence + C_robust * lipschitz_margin
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - offline env
    openai = None

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")


def generate_perturbations(text: str, n: int = 3) -> List[str]:
    """Generate semantic perturbations of input text.

    Uses the LLM to rewrite the text while preserving semantic meaning.
    Falls back to returning the original text if the LLM is unavailable.

    Parameters
    ----------
    text : str
        The input text to perturb.
    n : int, optional
        Number of semantic variations to generate. Default is 3.

    Returns
    -------
    List[str]
        List of n perturbed variations (or fallback of [text] if unavailable).
    """

    if not text or not isinstance(text, str):
        logger.warning("Invalid input text for perturbation; skipping")
        return [text] if text else [""]

    # Check if LLM is available
    runtime_key = os.environ.get("OPENAI_API_KEY")
    if not runtime_key or openai is None:
        logger.debug("Robustness Check Skipped: No OPENAI_API_KEY or openai module unavailable")
        return [text]

    # Construct the perturbation prompt
    system_prompt = (
        "You are a text rewriting expert. Your task is to rewrite sentences while preserving "
        "their exact semantic meaning. Use different words and structures, but do not change "
        "the core meaning or factual content."
    )
    user_prompt = (
        f"Rewrite the following text {n} times. Each rewrite should use different words and "
        f"structure but preserve the exact same meaning. Return only the rewritten versions, "
        f"one per line, without numbering or explanations.\n\nText: {text}"
    )

    try:
        api_delay = float(os.environ.get("ROBUSTNESS_API_DELAY", "0.05"))
        import time
        if api_delay > 0:
            time.sleep(api_delay)

        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=runtime_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                timeout=5.0,
            )
            response_text = resp.choices[0].message.content if resp and resp.choices else ""
        else:
            resp = openai.ChatCompletion.create(  # type: ignore[attr-defined]
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                timeout=5.0,
            )
            response_text = resp["choices"][0]["message"]["content"] if resp.get("choices") else ""

        if not response_text:
            logger.debug("Robustness Check Skipped: Empty LLM response")
            return [text]

        # Parse the response: split by newlines and filter empty lines
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        if not lines:
            logger.debug("Robustness Check Skipped: No valid perturbations from LLM")
            return [text]

        # Return up to n perturbations (may be fewer if LLM didn't produce enough)
        perturbations = lines[:n]
        if len(perturbations) < n:
            logger.debug(
                f"Requested {n} perturbations but got {len(perturbations)} from LLM; "
                "returning what we have"
            )

        logger.debug(f"Generated {len(perturbations)} perturbations for robustness check")
        return perturbations

    except (TimeoutError, ConnectionError, OSError) as exc:  # pragma: no cover - network call
        logger.debug(f"Robustness Check Skipped: Network error: {exc}")
        return [text]
    except AttributeError as exc:  # pragma: no cover - API version mismatch
        logger.debug(f"Robustness Check Skipped: OpenAI API compatibility issue: {exc}")
        return [text]
    except Exception as exc:  # pragma: no cover - catch-all for API errors
        logger.warning(f"Unexpected error in perturbation generation: {exc}", exc_info=True)
        return [text]


__all__ = [
    "generate_perturbations",
]
