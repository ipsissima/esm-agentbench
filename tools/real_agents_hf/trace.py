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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .agent_loop import AgentStep
from .embeddings import EmbeddingModel, embed_trace_steps
from .inference import ModelConfig

logger = logging.getLogger(__name__)


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

    for trace_file in sorted(trace_dir.glob("*.json")):
        try:
            with open(trace_file) as f:
                data = json.load(f)
                label = data.get('label')
                if label in traces:
                    traces[label].append(trace_file)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading {trace_file}: {e}")

    return traces


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
    traces = organize_traces_by_label(trace_dir)

    index = {
        'created': datetime.now().isoformat(),
        'trace_dir': str(trace_dir),
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
        **metadata,
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
