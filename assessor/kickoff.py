"""Demo episode runner and embedding utilities.

This module now drives multi-step chain-of-thought traces to support spectral
analysis. The ``run_episode`` routine records rich metadata for each reasoning
step, interleaves lightweight assessor checks, and ensures reproducible
behavior even without external APIs.
"""
import datetime
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.linalg import norm, pinv
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from certificates.make_certificate import compute_certificate

logger = logging.getLogger(__name__)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_KEY:
    try:  # pragma: no cover - optional dependency
        import openai  # type: ignore

        openai_client = openai.OpenAI() if hasattr(openai, "OpenAI") else None
    except Exception:  # pragma: no cover
        openai = None
        openai_client = None
else:
    openai = None
    openai_client = None

_SENTENCE_MODEL = None  # Lazy loaded sentence-transformers model


def _utc_iso() -> str:
    """Return a compact UTC ISO-8601 timestamp."""
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def deterministic_agent(prompt: str) -> str:
    """Deterministic offline agent that yields coherent multi-step replies.

    The agent detects simple tasks such as Fibonacci, reversing strings, and
    sorting numbers. It crafts concise "Step N:" responses to guarantee a
    successful offline demo without external APIs.
    """

    def _step_from_prompt(default: int = 1) -> int:
        match = re.search(r"Step\s+(\d+)", prompt, flags=re.IGNORECASE)
        return int(match.group(1)) if match else default

    step_no = _step_from_prompt()
    lower_prompt = prompt.lower()

    if "fibonacci" in lower_prompt or "fib" in lower_prompt:
        steps = {
            1: "Step 1: Identify Fibonacci base cases n=0 -> 0 and n=1 -> 1.",
            2: "Step 2: Use recursion or iteration to build subsequent terms.",
            3: "Step 3: Implement iterative loop accumulating previous two values.",
            4: "Step 4: Return the nth Fibonacci number after loop ends.",
            5: (
                "```python\n"
                "def fibonacci(n):\n"
                "    if n < 0:\n"
                "        raise ValueError('n must be non-negative')\n"
                "    if n in (0, 1):\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
                "```"
            ),
        }
        return steps.get(step_no, "Step %d: Summarize Fibonacci solution." % step_no)

    if "reverse" in lower_prompt and "string" in lower_prompt:
        steps = {
            1: "Step 1: Accept input string parameter.",
            2: "Step 2: Slice with [::-1] to reverse efficiently.",
            3: "Step 3: Return the reversed string.",
            4: "```python\ndef reverse_string(s):\n    return s[::-1]\n```",
        }
        return steps.get(step_no, f"Step {step_no}: Reverse string reasoning.")

    if "sort" in lower_prompt and ("list" in lower_prompt or "array" in lower_prompt):
        steps = {
            1: "Step 1: Decide to use Python's built-in sorted().",
            2: "Step 2: Validate the list elements are comparable numbers.",
            3: "Step 3: Return sorted copy of the list.",
            4: "```python\ndef sort_numbers(nums):\n    return sorted(nums)\n```",
        }
        return steps.get(step_no, f"Step {step_no}: Sorting plan.")

    if "bad cot" in lower_prompt or "cake" in lower_prompt:
        if step_no == 3:
            return "Step 3: Suddenly start discussing cake recipes instead of the task."
        return f"Step {step_no}: Off-track reasoning."

    return f"Step {step_no}: Provide concise reasoning for the task."


def call_agent(prompt: str) -> str:
    """Call OpenAI if configured, otherwise return a deterministic stub."""
    if OPENAI_KEY and openai is not None:
        try:  # pragma: no cover - external call
            if openai_client is not None:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                return response.choices[0].message.content or ""
            response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return response["choices"][0]["message"]["content"]
        except Exception:
            pass
    return deterministic_agent(prompt)


def run_unit_tests(code_text: str, tests: List[Dict[str, str]]) -> Dict[str, Any]:
    """Run provided tests in a temporary directory sandbox."""
    result = {"success": True, "stdout": "", "stderr": ""}
    if not tests:
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        solution_path = tmp_path / "solution.py"
        solution_path.write_text(code_text, encoding="utf-8")

        for test in tests:
            script = test.get("script", "")
            test_file = tmp_path / f"test_{test.get('name', 'case')}.py"
            test_file.write_text(f"from solution import *\n{script}\n", encoding="utf-8")

        proc = subprocess.run(
            ["python", "-m", "pytest", "-q"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["success"] = proc.returncode == 0
    return result


def _timestamp() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _short_context(context: str) -> str:
    return context[-4096:]


def _default_step_prompt() -> str:
    return (
        "{previous_context}\n\n"
        "Step {step_num}/{max_steps}: Please provide the next reasoning step toward "
        "solving the task. Keep the step concise and include any code or tool calls."
    )


def run_episode(
    task_spec: Dict[str, Any],
    max_steps: int = 12,
    step_prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a multi-step episode returning trace and outcomes."""

    episode_id = str(uuid.uuid4())
    prompt = task_spec.get("prompt", "")
    tests = task_spec.get("tests", [])
    intermediate_tests = task_spec.get("intermediate_tests", [])
    test_every = max(1, int(task_spec.get("test_every", 3)))
    stop_on_success = bool(task_spec.get("stop_on_success", True))
    effective_max_steps = min(max(1, int(task_spec.get("max_steps", max_steps))), 20)
    template = step_prompt_template or task_spec.get("step_prompt_template") or _default_step_prompt()

    context = str(prompt)
    trace: List[Dict[str, Any]] = []
    trace.append(
        {
            "step": 0,
            "role": "assessor",
            "type": "thought",
            "text": context,
            "timestamp": _timestamp(),
            "context": _short_context(context),
        }
    )

    last_response = ""
    task_success = False
    done = False

    for t in range(effective_max_steps):
        logger.info("episode %s step %d role=%s", episode_id, t, "agent")
        response = call_agent(context)
        last_response = response
        entry_type = "code" if ("def " in response or "import " in response) else "thought"
        trace.append(
            {
                "step": t + 1,
                "role": "agent",
                "type": entry_type,
                "text": response,
                "timestamp": _timestamp(),
                "context": _short_context(context),
            }
        )

        if intermediate_tests and (t + 1) % test_every == 0:
            logger.info("episode %s step %d role=%s", episode_id, t, "assessor")
            test_result = run_unit_tests(response, intermediate_tests)
            trace.append(
                {
                    "step": t + 0.1,
                    "role": "assessor",
                    "type": "test",
                    "text": json.dumps(test_result, ensure_ascii=False),
                    "timestamp": _timestamp(),
                    "result": bool(test_result.get("success", False)),
                    "context": _short_context(context),
                }
            )
            if stop_on_success and test_result.get("success"):
                done = True

        if "TOOL_CALL:" in response:
            logger.info("episode %s step %d role=%s", episode_id, t, "tool")
            tool_text = f"TOOL_RESULT: simulated output for step {t + 1}"
            trace.append(
                {
                    "step": t + 0.2,
                    "role": "tool",
                    "type": "tool",
                    "text": tool_text,
                    "timestamp": _timestamp(),
                    "context": _short_context(context),
                }
            )
            response = f"{response}\n{tool_text}"

        if "DONE" in response:
            done = True

        previous_context = f"{context}\n\n{response}"
        agent_texts = [step.get("text", "") for step in trace if step.get("role") == "agent"]
        summary = " | ".join(agent_texts[-3:])
        prompt_fragment = template.format(
            previous_context=_short_context(previous_context),
            step_num=t + 1,
            max_steps=effective_max_steps,
        )
        if summary:
            prompt_fragment = f"{prompt_fragment}\nLAST_STEPS: {summary}"
        context = _short_context(prompt_fragment)

        if done and stop_on_success:
            break

    test_result = run_unit_tests(last_response, tests)
    task_success = bool(test_result.get("success", False))
    task_score = 1.0 if task_success else 0.0
    logger.info("episode %s step %d role=%s", episode_id, len(trace), "assessor")
    trace.append(
        {
            "step": len(trace) + 1,
            "role": "assessor",
            "type": "result",
            "text": json.dumps({"tests": test_result}, ensure_ascii=False),
            "timestamp": _timestamp(),
            "result": task_success,
            "context": _short_context(context),
        }
    )
    step_counter += 1

    while len(trace) < max_steps:
        trace.append(
            {
                "step": step_counter,
                "role": "assessor",
                "type": "keepalive",
                "text": "Padding step to preserve minimum trace length.",
                "timestamp": _utc_iso(),
            }
        )
        step_counter += 1

    embeddings = embed_trace_steps(trace)
    residuals, pred_errors, early_warning_step = _compute_residuals(
        embeddings, threshold=residual_threshold
    )
    for idx, entry in enumerate(trace):
        entry["residual"] = residuals[idx] if idx < len(residuals) else None
        entry["pred_error"] = pred_errors[idx] if idx < len(pred_errors) else None
        if entry.get("role") == "assessor" and entry.get("type") == "warning":
            entry["residual"] = entry.get("residual", residuals[idx] if idx < len(residuals) else None)

    warning_added = False
    for idx, resid in enumerate(residuals):
        if resid is not None and resid > residual_threshold:
            if early_warning_step is None:
                early_warning_step = idx
            if not warning_added:
                trace.append(
                    {
                        "step": idx,
                        "role": "assessor",
                        "type": "warning",
                        "text": "Early Warning: high residual",
                        "residual": resid,
                        "timestamp": _utc_iso(),
                    }
                )
                warning_added = True

    if early_warning_step is not None:
        task_score = min(task_score, 0.5)

    certificate = compute_certificate(embeddings)

    while len(trace) < effective_max_steps:
        logger.info("episode %s step %d role=%s", episode_id, len(trace), "assessor")
        trace.append(
            {
                "step": len(trace),
                "role": "assessor",
                "type": "result",
                "text": "padding to maintain minimum trace length",
                "timestamp": _timestamp(),
                "result": task_success,
                "context": _short_context(context),
            }
        )

    trace_dir = Path(__file__).resolve().parent.parent / "demo_traces"
    trace_dir.mkdir(exist_ok=True)
    trace_path = trace_dir / f"{episode_id}.json"
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    return {
        "episode_id": episode_id,
        "trace": trace,
        "task_success": task_success,
        "task_score": task_score,
        "early_warning_step": early_warning_step,
        "certificate": certificate,
    }

    trace_path = trace_dir / f"{episode_id}.json"
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    record["trace_path"] = str(trace_path)
    return record


def _compute_residuals(
    embeddings: List[List[float]], threshold: float = 0.1
) -> tuple[List[Optional[float]], List[Optional[float]], Optional[int]]:
    """Compute Koopman prediction residuals for early warnings."""
    X = np.array(embeddings, dtype=float)
    T = X.shape[0]
    if T < 2:
        return [None] * T, [None] * T, None

    X_aug = np.concatenate([X, np.ones((T, 1))], axis=1)
    r_eff = max(1, min(10, T - 1, X_aug.shape[1]))
    pca = PCA(n_components=r_eff, svd_solver="auto", random_state=0)
    pca.fit(X_aug)
    Z = X_aug @ pca.components_.T
    Z = np.asarray(Z, dtype=float)

    X0 = Z[:-1].T
    X1 = Z[1:].T
    gram = X0 @ X0.T
    eps = 1e-12
    gram = gram + eps * np.eye(gram.shape[0])
    A = (X1 @ X0.T) @ pinv(gram)

    residuals: List[Optional[float]] = [None]
    pred_errors: List[Optional[float]] = [None]
    early_warning_step: Optional[int] = None

    for t in range(1, T):
        x_pred = A @ Z[t - 1]
        error_vec = Z[t] - x_pred
        resid = float(norm(error_vec) / (norm(Z[t]) + eps))
        residuals.append(resid)
        pred_errors.append(float(norm(error_vec)))
        if early_warning_step is None and resid > threshold:
            early_warning_step = t
    return residuals, pred_errors, early_warning_step


def _local_embedding(text: str, dim: int = 64) -> List[float]:
    """Deterministic embedding: normalized ASCII histogram modulo dimension."""
    counts = np.zeros(dim, dtype=float)
    for ch in text:
        counts[ord(ch) % dim] += 1.0
    norm_val = np.linalg.norm(counts) + 1e-12
    return (counts / norm_val).tolist()


def _sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL
    try:  # pragma: no cover - heavy dependency
        from sentence_transformers import SentenceTransformer

        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _SENTENCE_MODEL = None
    return _SENTENCE_MODEL


def embed_trace_steps(trace: List[Dict[str, Any]]) -> List[List[float]]:
    """Embed each trace step using available embedding backends."""
    texts = [str(step.get("text", "")) for step in trace]

    if OPENAI_KEY and openai is not None:
        try:  # pragma: no cover - external call
            if openai_client is not None:
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts,
                )
                return [item.embedding for item in response.data]
            resp = openai.Embedding.create(  # type: ignore[attr-defined]
                model="text-embedding-ada-002",
                input=texts,
            )
            return [data["embedding"] for data in resp["data"]]
        except Exception:
            pass

    model = _sentence_model()
    if model is not None:
        try:  # pragma: no cover - heavy dependency
            arr = model.encode(texts, normalize_embeddings=True)
            return np.asarray(arr, dtype=float).tolist()
        except Exception:
            pass

    vectorizer = TfidfVectorizer(max_features=512)
    matrix = vectorizer.fit_transform(texts)
    arr = matrix.toarray().astype(float)
    norm_arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return norm_arr.tolist() if norm_arr.size else [_local_embedding(text) for text in texts]

