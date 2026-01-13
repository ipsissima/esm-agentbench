#!/usr/bin/env python3
"""Trace recording and storage for real agent runs.

Records:
- Model configuration and run metadata
- Step-by-step agent execution
- Embeddings for each step
- Final outcome and success metrics
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .agent_loop import AgentStep
from .embeddings import EmbeddingModel, embed_trace_steps
from .inference import ModelConfig

logger = logging.getLogger(__name__)
RUN_GENERATOR = "tools/real_agents_hf/run_real_agents.py"


@dataclass
class RunMetadata:
    """Metadata for a single agent run."""
    run_id: str
    scenario: str
    model_name: str
    model_hf_id: str
    backend: str
    label: str  # 'gold', 'creative', or 'drift'
    timestamp: str
    prompt: str
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'scenario': self.scenario,
            'model': {
                'name': self.model_name,
                'hf_id': self.model_hf_id,
                'backend': self.backend,
            },
            'label': self.label,
            'timestamp': self.timestamp,
            'prompt': self.prompt,
            'seed': self.seed,
        }


class TraceRecorder:
    """Records and saves agent execution traces."""

    def __init__(self, output_dir: Path, embedding_model: EmbeddingModel):
        """Initialize trace recorder.

        Parameters
        ----------
        output_dir : Path
            Directory to save traces
        embedding_model : EmbeddingModel
            Loaded embedding model for computing step embeddings
        """
        self.output_dir = output_dir
        self.embedding_model = embedding_model
        output_dir.mkdir(parents=True, exist_ok=True)

    def save_trace(
        self,
        metadata: RunMetadata,
        steps: List[AgentStep],
        outcome: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save trace to disk.

        Parameters
        ----------
        metadata : RunMetadata
            Run metadata
        steps : list of AgentStep
            Agent execution steps
        outcome : dict, optional
            Outcome metrics (e.g., tests passed, task completed)

        Returns
        -------
        Path
            Path to saved trace file
        """
        # Convert steps to dicts
        steps_dict = [step.to_dict() for step in steps]

        # Compute embeddings for steps
        logger.info(f"Computing embeddings for {len(steps_dict)} steps...")
        embeddings = embed_trace_steps(steps_dict, self.embedding_model)

        # Add embeddings to steps
        for step_dict, emb in zip(steps_dict, embeddings):
            step_dict['embedding'] = emb.tolist()

        # Build trace
        trace = metadata.to_dict()
        trace['steps'] = steps_dict
        trace['embeddings'] = [emb.tolist() for emb in embeddings]

        if outcome:
            trace['outcome'] = outcome

        # Save to file
        trace_file = self.output_dir / f"{metadata.run_id}.json"
        with open(trace_file, 'w') as f:
            json.dump(trace, f, indent=2)

        logger.info(f"Saved trace to {trace_file}")
        return trace_file

    def load_trace(self, trace_file: Path) -> Dict[str, Any]:
        """Load trace from file.

        Parameters
        ----------
        trace_file : Path
            Path to trace file

        Returns
        -------
        dict
            Trace data
        """
        with open(trace_file) as f:
            return json.load(f)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def create_run_id(scenario: str, model_name: str, label: str, run_num: int) -> str:
    """Create unique run ID.

    Parameters
    ----------
    scenario : str
        Scenario name
    model_name : str
        Model name
    label : str
        Label (gold/creative/drift)
    run_num : int
        Run number

    Returns
    -------
    str
        Run ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{scenario}_{model_name}_{label}_{run_num:03d}_{timestamp}"


def organize_traces_by_label(trace_dir: Path) -> Dict[str, List[Path]]:
    """Organize trace files by label.

    Parameters
    ----------
    trace_dir : Path
        Directory containing trace JSON files

    Returns
    -------
    dict
        Mapping from label to list of trace files
    """
    traces = {'gold': [], 'creative': [], 'drift': []}

    for trace_file in sorted(trace_dir.glob("**/*.json")):
        if trace_file.name == "index.json":
            continue
        try:
            with open(trace_file) as f:
                data = json.load(f)
                label = data.get('label')
                if label in traces:
                    traces[label].append(trace_file)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading {trace_file}: {e}")

    return traces


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_trace_entries(trace_dir: Path) -> List[Dict[str, Any]]:
    """Collect trace metadata and hashes for attestation."""
    entries: List[Dict[str, Any]] = []
    for trace_file in sorted(trace_dir.glob("**/*.json")):
        if trace_file.name == "index.json":
            continue
        try:
            raw = trace_file.read_bytes()
            digest = _compute_sha256(raw)
            data = json.loads(raw.decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Error reading {trace_file}: {exc}")
            continue

        label = data.get("label")
        entries.append(
            {
                "file": str(trace_file.relative_to(trace_dir)),
                "sha256": digest,
                "size_bytes": len(raw),
                "label": label,
                "timestamp": data.get("timestamp"),
                "run_id": data.get("run_id"),
                "model": data.get("model", {}).get("name"),
            }
        )

    return entries


def build_hash_chain(entries: List[Dict[str, Any]]) -> List[str]:
    """Build a simple hash chain across trace hashes."""
    chain: List[str] = []
    prev = "genesis"
    for entry in sorted(entries, key=lambda e: e["file"]):
        payload = f"{prev}:{entry['sha256']}:{entry['file']}".encode("utf-8")
        prev = _compute_sha256(payload)
        chain.append(prev)
    return chain


def create_index(trace_dir: Path, metadata: Dict[str, Any]) -> Path:
    """Create index.json for trace directory.

    Parameters
    ----------
    trace_dir : Path
        Directory containing traces
    metadata : dict
        Additional metadata (counts, runtime, etc.)

    Returns
    -------
    Path
        Path to index file
    """
    safe_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in ("run_generator", "attestation")
    }
    traces = organize_traces_by_label(trace_dir)
    trace_entries = collect_trace_entries(trace_dir)
    hash_chain = build_hash_chain(trace_entries)
    run_signature_payload = {
        "created": datetime.now().isoformat(),
        "run_generator": metadata.get("run_generator", RUN_GENERATOR),
        "hash_chain_tail": hash_chain[-1] if hash_chain else "genesis",
    }
    run_signature = _compute_sha256(
        json.dumps(run_signature_payload, sort_keys=True).encode("utf-8")
    )

    index = {
        'created': run_signature_payload["created"],
        'trace_dir': str(trace_dir),
        'run_generator': run_signature_payload["run_generator"],
        'counts': {
            'gold': len(traces['gold']),
            'creative': len(traces['creative']),
            'drift': len(traces['drift']),
            'total': sum(len(v) for v in traces.values()),
        },
        'traces_by_label': {
            label: [str(f.name) for f in files]
            for label, files in traces.items()
        },
        'attestation': {
            'algorithm': 'sha256',
            'trace_hashes': trace_entries,
            'hash_chain': hash_chain,
            'run_signature': run_signature,
        },
        **safe_metadata,
    }

    index_file = trace_dir / "index.json"
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Created index at {index_file}")
    return index_file


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test trace recording
    from .embeddings import EmbeddingModel

    output_dir = Path("/tmp/test_traces")
    output_dir.mkdir(exist_ok=True)

    # Load embedding model
    emb_model = EmbeddingModel()
    emb_model.load()

    # Create test trace
    recorder = TraceRecorder(output_dir, emb_model)

    metadata = RunMetadata(
        run_id="test_001",
        scenario="test_scenario",
        model_name="test_model",
        model_hf_id="test/model",
        backend="transformers",
        label="gold",
        timestamp=datetime.now().isoformat(),
        prompt="Test task",
    )

    steps = [
        AgentStep(0, 'thought', 'I need to test this'),
        AgentStep(1, 'tool_call', '{"tool":"list_dir","args":{}}',
                  tool_name='list_dir', tool_args={}),
        AgentStep(2, 'tool_result', 'file1.py\nfile2.py'),
        AgentStep(3, 'final', 'Task complete'),
    ]

    trace_file = recorder.save_trace(metadata, steps)
    print(f"Saved trace: {trace_file}")

    # Create index
    index_file = create_index(output_dir, {'test_run': True})
    print(f"Created index: {index_file}")

    emb_model.unload()
