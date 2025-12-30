# Real Agent HF Evaluation

Infrastructure for running coding agents using local Hugging Face models.

## Components

### Core Modules

- **`models.yaml`** - Model configuration (HF IDs, backends, parameters)
- **`inference.py`** - Inference backends (transformers, vLLM with auto-fallback)
- **`sandbox.py`** - Safe execution environment with path confinement
- **`tools.py`** - Agent tools (read_file, write_file, run_cmd, git_diff, grep)
- **`agent_loop.py`** - Agent execution with JSON tool protocol
- **`embeddings.py`** - Local embedding generation with caching
- **`trace.py`** - Trace recording and storage
- **`run_real_agents.py`** - Main CLI runner

### Tool Protocol

Agents use a strict JSON protocol (no native function calling required):

```
THOUGHT: I need to check what files exist.

TOOL_CALL:
{"tool": "list_dir", "args": {"path": "."}}

(system returns result)

TOOL_CALL:
{"tool": "read_file", "args": {"path": "README.md"}}

(system returns result)

FINAL: The project contains...
```

### Available Tools

1. **read_file(path)** - Read file contents
2. **write_file(path, content)** - Write file
3. **list_dir(path)** - List directory contents
4. **run_cmd(cmd: list)** - Execute shell command (sandboxed)
5. **git_diff()** - Show git diff
6. **grep(pattern, path)** - Search for pattern

## Quick Start

```bash
# 1. Run one scenario with default model
python run_real_agents.py \
  --scenario supply_chain_poisoning \
  --n 5

# 2. Run all scenarios with multiple models
python run_real_agents.py \
  --all-scenarios \
  --models deepseek-coder-7b-instruct,codellama-13b-instruct \
  --n 20

# 3. Validate results
cd ../../
python analysis/run_real_hf_experiment.py \
  --all-scenarios \
  --cross-model
```

## Configuration

### Adding a New Model

Edit `models.yaml`:

```yaml
models:
  - name: my-model
    hf_id: organization/model-name
    backend: transformers  # or 'vllm'
    dtype: bfloat16
    max_tokens: 2048
    temperature: 0.2
    load_in_4bit: true  # optional, for quantization
```

### Backend Selection

- **transformers**: Universal, CPU/GPU, supports quantization
- **vllm**: Fast, GPU-only (Linux + CUDA), automatic batching

If vLLM is not available, automatically falls back to transformers.

## Architecture

```
run_real_agents.py
    │
    ├─► models.yaml ─► inference.py (Backend)
    │                      │
    │                      ├─► TransformersBackend
    │                      └─► VLLMBackend
    │
    ├─► Sandbox (scenario workspace)
    │      │
    │      └─► Tools (file ops, commands)
    │
    ├─► AgentLoop (execution + protocol)
    │      │
    │      └─► Steps (THOUGHT/TOOL_CALL/FINAL)
    │
    ├─► EmbeddingModel (local, cached)
    │
    └─► TraceRecorder
           │
           └─► Output: {model}/{label}/run_NNN.json
```

## Output Format

Each trace is saved as JSON:

```json
{
  "run_id": "scenario_model_label_001_20250101_120000",
  "scenario": "supply_chain_poisoning",
  "model": {
    "name": "deepseek-coder-7b-instruct",
    "hf_id": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "backend": "vllm"
  },
  "label": "gold",
  "timestamp": "2025-01-01T12:00:00",
  "prompt": "Task: Recommend a safe Python HTTP library...",
  "steps": [
    {
      "i": 0,
      "type": "thought",
      "text": "I need to read the package registry",
      "embedding": [0.123, -0.456, ...]
    },
    {
      "i": 1,
      "type": "tool_call",
      "tool": "read_file",
      "args": {"path": "targets/package_registry.py"},
      "text": "{\"tool\":\"read_file\",...}",
      "embedding": [...]
    },
    {
      "i": 2,
      "type": "tool_result",
      "text": "File contents: ...",
      "embedding": [...]
    },
    {
      "i": 3,
      "type": "final",
      "text": "Recommended package: requests",
      "embedding": [...]
    }
  ],
  "embeddings": [[...], [...], ...],
  "outcome": {
    "num_steps": 4,
    "elapsed_seconds": 12.3,
    "completed": true,
    "error": false
  }
}
```

## Troubleshooting

### Out of Memory

Use quantization or smaller model:

```yaml
# In models.yaml
load_in_4bit: true
```

Or use a smaller model like `phi-3-mini-instruct`.

### vLLM Not Available

Automatically falls back to transformers. No action needed.

### Agent Gets Stuck

Reduce `--max-steps`:

```bash
python run_real_agents.py --max-steps 20 ...
```

### Models Not Downloading

Check internet connection and HuggingFace Hub access. Pre-download:

```python
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-7b-instruct-v1.5")
```

## Testing

```bash
# Test with tiny model (fast)
python run_real_agents.py \
  --scenario supply_chain_poisoning \
  --models tiny-test \
  --n 2 \
  --max-steps 10

# Verify output
ls -la ../../submissions/ipsissima/supply_chain_poisoning/experiment_traces_real_hf/
```

## See Also

- Main documentation: `../../docs/REAL_AGENT_HF_EVAL.md`
- Spectral validation: `../../analysis/run_real_hf_experiment.py`
- Scenario prompts: `../../scenarios/{scenario}/real_agent_prompts/`
