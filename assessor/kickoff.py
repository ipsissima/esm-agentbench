"""Demo episode runner and embedding utilities."""
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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
    # Fallback deterministic stub
    return "# solution\ndef solution():\n    return 42\n"


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


def run_episode(task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a minimal episode returning trace and outcomes."""
    episode_id = str(uuid.uuid4())
    prompt = task_spec.get("prompt", "")
    tests = task_spec.get("tests", [])

    trace: List[Dict[str, Any]] = []
    trace.append({"step": 0, "role": "assessor", "text": prompt, "tool_calls": []})

    agent_output = call_agent(prompt)
    trace.append({"step": 1, "role": "agent", "text": agent_output, "tool_calls": []})

    test_result = run_unit_tests(agent_output, tests)
    task_success = bool(test_result.get("success", False))
    task_score = 1.0 if task_success else 0.0

    trace.append(
        {
            "step": 2,
            "role": "assessor",
            "text": json.dumps({"tests": test_result}, ensure_ascii=False),
            "tool_calls": [],
        }
    )

    return {
        "episode_id": episode_id,
        "trace": trace,
        "task_success": task_success,
        "task_score": task_score,
    }


def _local_embedding(text: str, dim: int = 64) -> List[float]:
    """Deterministic embedding: normalized ASCII histogram modulo dimension."""
    counts = np.zeros(dim, dtype=float)
    for ch in text:
        counts[ord(ch) % dim] += 1.0
    norm = np.linalg.norm(counts) + 1e-12
    return (counts / norm).tolist()


def embed_trace_steps(trace: List[Dict[str, Any]]) -> List[List[float]]:
    """Embed each trace step using OpenAI if available otherwise local histogram."""
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

    return [_local_embedding(text) for text in texts]
