"""Episode runner with chain-of-thought tracing and coherence analysis."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

np.random.seed(0)


def _utc_iso() -> str:
    """Return a compact UTC ISO-8601 timestamp."""

    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _short_context(text: str, limit: int = 4096) -> str:
    return text[-limit:]


def _split_steps(response: str) -> List[str]:
    """Split multi-step agent replies into separate step segments."""

    pattern = re.compile(r"(?:^|\n)(Step\s*\d+\s*:)", flags=re.IGNORECASE)
    matches = list(pattern.finditer(response))
    if not matches:
        return [response.strip()]

    segments = []
    for idx, match in enumerate(matches):
        start = match.start(1)
        end = matches[idx + 1].start(1) if idx + 1 < len(matches) else len(response)
        seg = response[start:end].strip()
        if seg:
            segments.append(seg)
    return segments or [response.strip()]


def _extract_code(text: str) -> Optional[str]:
    """Extract the last fenced code block from text."""

    fence = re.findall(r"```(?:python)?\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        return fence[-1]
    return None


def deterministic_agent(prompt: str) -> str:
    """Deterministic offline agent producing reproducible CoT steps."""

    lower = prompt.lower()
    step_match = re.search(r"step\s*(\d+)", prompt, flags=re.IGNORECASE)
    step_no = int(step_match.group(1)) if step_match else 1

    if "fibonacci" in lower or "fib" in lower:
        steps = {
            1: "Step 1: Define Fibonacci base cases n=0->0 and n=1->1.",
            2: "Step 2: Iterate from 2..n accumulating previous two numbers.",
            3: "Step 3: Maintain (a,b) and update (b,a+b) each loop.",
            4: "Step 4: Return b after finishing the loop.",
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
        return steps.get(step_no, f"Step {step_no}: Summarize Fibonacci solution.")

    if "reverse" in lower and "string" in lower:
        steps = {
            1: "Step 1: Accept the input string parameter.",
            2: "Step 2: Reverse using slicing s[::-1] for O(n).",
            3: "Step 3: Return the reversed string.",
            4: "```python\n"
            "def reverse_string(s):\n"
            "    return s[::-1]\n"
            "```",
        }
        return steps.get(step_no, f"Step {step_no}: Reverse string reasoning.")

    if "palindrome" in lower:
        steps = {
            1: "Step 1: Normalize the string to compare ends.",
            2: "Step 2: Use two-pointer scan or slicing.",
            3: "Step 3: Equality between s and s[::-1] decides palindrome.",
            4: "```python\n"
            "def is_palindrome(s):\n"
            "    cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
            "    return cleaned == cleaned[::-1]\n"
            "```",
        }
        return steps.get(step_no, f"Step {step_no}: Palindrome check.")

    if "sort" in lower and ("list" in lower or "array" in lower or "numbers" in lower):
        steps = {
            1: "Step 1: Validate input list of comparable numbers.",
            2: "Step 2: Use Python's built-in sorted for stability.",
            3: "Step 3: Return the sorted copy.",
            4: "```python\n"
            "def sort_numbers(nums):\n"
            "    return sorted(nums)\n"
            "```",
        }
        return steps.get(step_no, f"Step {step_no}: Sorting plan.")

    if "bad cot" in lower or "hallucination" in lower:
        if step_no == 3:
            return "Step 3: Suddenly start discussing cake recipes instead of the task."
        return f"Step {step_no}: Off-track reasoning."

    return f"Step {step_no}: Provide concise reasoning for the task."


def call_agent(prompt: str) -> str:
    """Call OpenAI if configured, otherwise return deterministic output."""

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
            logger.warning("OpenAI call failed; using deterministic agent.")
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
            name = test.get("name", "case")
            test_file = tmp_path / f"test_{name}.py"
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


def _default_step_prompt() -> str:
    return (
        "{previous_context}\n\n"
        "Step {step_num}/{max_steps}: Provide the next reasoning step."
        " Keep it concise and include any code or tool calls."
    )


def embed_trace_steps(trace: List[Dict[str, Any]]) -> np.ndarray:
    """Embed each trace step using available embedding backends.

    Preference order: OpenAI embeddings -> sentence-transformers -> TF-IDF ->
    deterministic histogram fallback. Embeddings are L2 normalized.
    """

    texts = [str(step.get("text", "")) for step in trace]

    if OPENAI_KEY and openai is not None:
        try:  # pragma: no cover - external call
            if openai_client is not None:
                response = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts,
                )
                arr = np.array([item.embedding for item in response.data], dtype=float)
            else:
                resp = openai.Embedding.create(  # type: ignore[attr-defined]
                    model="text-embedding-ada-002",
                    input=texts,
                )
                arr = np.array([data["embedding"] for data in resp["data"]], dtype=float)
            arr = arr / (norm(arr, axis=1, keepdims=True) + 1e-12)
            return arr
        except Exception:
            logger.warning("OpenAI embedding failed; falling back to offline model.")

    model = _sentence_model()
    if model is not None:
        try:  # pragma: no cover - heavy dependency
            arr = model.encode(texts, normalize_embeddings=True)
            return np.asarray(arr, dtype=float)
        except Exception:
            logger.warning("Sentence-transformers embedding failed; using TF-IDF.")

    vectorizer = TfidfVectorizer(max_features=512)
    matrix = vectorizer.fit_transform(texts)
    arr = matrix.toarray().astype(float)
    if arr.size:
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr

    # Fallback deterministic histogram
    return np.array([_local_embedding(texts[0] if texts else "") for _ in texts], dtype=float)


def _sentence_model() -> Any:
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL
    try:  # pragma: no cover - heavy dependency
        from sentence_transformers import SentenceTransformer

        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _SENTENCE_MODEL = None
    return _SENTENCE_MODEL


def _local_embedding(text: str, dim: int = 64) -> List[float]:
    counts = np.zeros(dim, dtype=float)
    for ch in text:
        counts[ord(ch) % dim] += 1.0
    norm_val = np.linalg.norm(counts) + 1e-12
    return (counts / norm_val).tolist()


def _compute_residuals(
    embeddings: np.ndarray, threshold: float
) -> Tuple[List[Optional[float]], List[Optional[float]], Optional[int]]:
    """Compute Koopman residuals and earliest warning step."""

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)
    T = embeddings.shape[0]
    if T < 2:
        return [None] * T, [None] * T, None

    X_aug = np.concatenate([embeddings, np.ones((T, 1))], axis=1)
    r_eff = max(1, min(10, T - 1, X_aug.shape[1]))
    pca = PCA(n_components=r_eff, svd_solver="auto", random_state=0)
    pca.fit(X_aug)
    Z = X_aug @ pca.components_.T

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


def run_episode(
    task_spec: Dict[str, Any],
    max_steps: Optional[int] = 12,
    step_prompt_template: Optional[str] = None,
    check_every: int = 3,
    stop_on_success: bool = True,
) -> Dict[str, Any]:
    """Execute a multi-step episode returning trace and outcomes."""

    episode_id = str(uuid.uuid4())
    prompt = task_spec.get("prompt", "")
    tests = task_spec.get("tests", [])
    intermediate_tests = task_spec.get("intermediate_tests", [])
    test_every = max(1, int(task_spec.get("test_every", check_every)))
    stop_flag = bool(task_spec.get("stop_on_success", stop_on_success))
    residual_threshold = float(task_spec.get("residual_threshold", 0.1))
    effective_max_steps = max(1, min(int(task_spec.get("max_steps", max_steps or 12)), 20))
    template = step_prompt_template or task_spec.get("step_prompt_template") or _default_step_prompt()

    history = str(prompt)
    trace: List[Dict[str, Any]] = []
    trace.append(
        {
            "step": 0,
            "role": "assessor",
            "type": "thought",
            "text": history,
            "timestamp": _utc_iso(),
            "context": _short_context(history),
        }
    )

    candidate_code = ""
    last_response = ""
    task_success = False

    for step_idx in range(1, effective_max_steps + 1):
        step_prompt = template.format(
            previous_context=_short_context(history),
            step_num=step_idx,
            max_steps=effective_max_steps,
        )
        response = call_agent(step_prompt)
        last_response = response
        segments = _split_steps(response)

        for seg_i, seg in enumerate(segments):
            entry_step = step_idx + seg_i * 0.01
            entry_type = "code" if "def " in seg or "```" in seg else "thought"
            trace.append(
                {
                    "step": entry_step,
                    "role": "agent",
                    "type": entry_type,
                    "text": seg,
                    "timestamp": _utc_iso(),
                    "context": _short_context(history),
                }
            )
            code_block = _extract_code(seg)
            if code_block:
                candidate_code = code_block
        history = _short_context(f"{history}\n\n{response}")

        if intermediate_tests and (step_idx % test_every == 0):
            test_result = run_unit_tests(candidate_code or last_response, intermediate_tests)
            trace.append(
                {
                    "step": step_idx + 0.1,
                    "role": "assessor",
                    "type": "test",
                    "text": json.dumps(test_result, ensure_ascii=False),
                    "timestamp": _utc_iso(),
                    "result": bool(test_result.get("success", False)),
                    "context": _short_context(history),
                }
            )
            if stop_flag and test_result.get("success"):
                task_success = True
                break

    final_test = run_unit_tests(candidate_code or last_response, tests)
    task_success = bool(final_test.get("success", False)) or task_success
    trace.append(
        {
            "step": len(trace) + 1,
            "role": "assessor",
            "type": "result",
            "text": json.dumps({"tests": final_test}, ensure_ascii=False),
            "timestamp": _utc_iso(),
            "result": task_success,
            "context": _short_context(history),
        }
    )

    while len(trace) < effective_max_steps:
        trace.append(
            {
                "step": len(trace),
                "role": "assessor",
                "type": "keepalive",
                "text": "Padding step to preserve minimum trace length.",
                "timestamp": _utc_iso(),
                "context": _short_context(history),
            }
        )

    embeddings = embed_trace_steps(trace)
    residuals, pred_errors, early_warning_step = _compute_residuals(embeddings, residual_threshold)

    for idx, entry in enumerate(trace):
        entry["residual"] = residuals[idx] if idx < len(residuals) else None
        entry["pred_error"] = pred_errors[idx] if idx < len(pred_errors) else None

    warning_added = False
    for idx, resid in enumerate(residuals):
        if resid is not None and resid > residual_threshold:
            if not warning_added:
                trace.append(
                    {
                        "step": float(idx),
                        "role": "assessor",
                        "type": "warning",
                        "text": f"Early Warning: residual {resid:.3f} exceeds threshold {residual_threshold}",
                        "timestamp": _utc_iso(),
                        "context": _short_context(history),
                        "residual": resid,
                    }
                )
                warning_added = True
            if early_warning_step is None:
                early_warning_step = idx

    certificate = compute_certificate(embeddings)
    certificate["per_step_residuals"] = residuals

    trace_dir = Path(__file__).resolve().parent.parent / "demo_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{episode_id}.json"
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    return {
        "episode_id": episode_id,
        "trace": trace,
        "trace_path": str(trace_path),
        "task_success": task_success,
        "task_score": 1.0 if task_success else 0.0,
        "early_warning_step": early_warning_step,
        "per_step_residuals": residuals,
        "certificate": certificate,
    }


__all__ = [
    "run_episode",
    "embed_trace_steps",
    "run_unit_tests",
    "call_agent",
    "deterministic_agent",
]
