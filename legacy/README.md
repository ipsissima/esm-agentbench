# Legacy / Deprecated Code

This directory contains deprecated code that is **NOT** used for benchmark evaluation.

These files are retained only for:
- Historical reference
- Legacy test compatibility (CI only)
- Backward compatibility with older tooling

## Contents

### `convert_trace.py`
- **Status**: DEPRECATED (moved from `analysis/`)
- **Reason**: Synthetic trace generation is no longer supported.
- **Replacement**: `tools/real_agents_hf/run_real_agents.py`
- **Warning**: Do NOT use for benchmark submissions.

### `generate_seed_traces.py`
- **Status**: DEPRECATED (moved from `tools/`)
- **Reason**: Synthetic trace generation is no longer supported.
- **Replacement**: `tools/real_agents_hf/run_real_agents.py`
- **Warning**: Do NOT use for benchmark submissions.

## Important

**Do not use any code from this directory for benchmark submissions.**

All Phase-1 evaluation must use real agent traces from:
- `tools/real_agents_hf/run_real_agents.py` - Generate traces with real HuggingFace models
- `analysis/run_real_hf_experiment.py` - Analyze real agent traces

## Migration Guide

If you have code using deprecated scripts:

```python
# OLD (Deprecated - do NOT use):
from analysis.convert_trace import generate_synthetic_traces
from legacy.convert_trace import generate_synthetic_traces

# NEW (Required for submissions):
from tools.real_agents_hf.run_real_agents import run_real_agent_evaluation
```

For more information, see:
- `docs/REAL_AGENT_HF_EVAL.md` - Real agent evaluation guide
- `tools/real_agents_hf/README.md` - Real agent infrastructure docs
