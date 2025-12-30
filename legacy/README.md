# Legacy / Deprecated Code

This directory contains references to deprecated code that is kept for backward compatibility only.

## Deprecated Scripts

The following scripts are **DEPRECATED** and should not be used for new work:

### 1. `analysis/convert_trace.py`
- **Status**: DEPRECATED
- **Reason**: Synthetic trace generation is no longer supported. Use real agent evaluation instead.
- **Replacement**: `tools/real_agents_hf/run_real_agents.py`
- **Note**: Kept for backward compatibility with legacy tests only.

### 2. `tools/generate_seed_traces.py`
- **Status**: DEPRECATED
- **Reason**: Synthetic trace generation is no longer supported.
- **Replacement**: `tools/real_agents_hf/run_real_agents.py`
- **Note**: Use real agent evaluation for all new work.

## Migration Guide

If you have code using deprecated scripts:

```python
# OLD (Deprecated):
from analysis.convert_trace import generate_synthetic_traces

# NEW (Recommended):
from tools.real_agents_hf.run_real_agents import run_real_agent_evaluation
```

For more information, see:
- `docs/REAL_AGENT_HF_EVAL.md` - Real agent evaluation guide
- `tools/real_agents_hf/README.md` - Real agent infrastructure docs
