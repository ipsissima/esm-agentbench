import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import assessor.kickoff as kickoff


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    def __init__(self, content: str):
        self._content = content
        self.chat = self
        self.completions = self

    def create(self, *args, **kwargs):
        return _FakeResp(self._content)


@pytest.fixture(autouse=True)
def reset_oracle_env(monkeypatch):
    monkeypatch.delenv("SEMANTIC_ORACLE_MODEL", raising=False)
    monkeypatch.delenv("ORACLE_API_DELAY", raising=False)
    monkeypatch.setattr(kickoff, "_ORACLE_DEFAULT_TIMEOUT", 5.0)
    yield


def test_semantic_oracle_flags_warning(monkeypatch):
    verdict_json = "{" "\"safe\": false, \"reason\": \"rm -rf\", \"confidence\": 0.98" "}"
    fake_openai = types.SimpleNamespace(OpenAI=lambda api_key=None: _FakeClient(verdict_json))
    monkeypatch.setattr(kickoff, "openai", fake_openai)
    monkeypatch.setattr(kickoff, "OPENAI_KEY", "sk-test")
    monkeypatch.setenv("ORACLE_API_DELAY", "0")
    monkeypatch.setattr(kickoff, "call_agent", lambda prompt: "Step 1: run rm -rf /", raising=False)

    result = kickoff.run_episode({"prompt": "test semantic oracle"}, max_steps=1)

    trace = result["trace"]
    agent_entries = [step for step in trace if step.get("role") == "agent"]
    assert agent_entries, "Agent steps should exist"
    assert agent_entries[0].get("oracle_safe") is False
    assert agent_entries[0].get("oracle_reason") == "rm -rf"
    warnings = [step for step in trace if step.get("type") == "warning" and "Semantic Oracle" in step.get("text", "")]
    assert warnings, "Semantic oracle warning entry should be appended"
    assert result.get("semantic_warning_step") == pytest.approx(1.0)


def test_semantic_oracle_offline(monkeypatch):
    monkeypatch.setattr(kickoff, "OPENAI_KEY", None)
    monkeypatch.setattr(kickoff, "openai", None)

    verdict = kickoff.semantic_sanity_check("context", "action")

    assert verdict["safe"] is True
    assert "Offline mode" in verdict["reason"]
