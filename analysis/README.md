# Analysis Scripts

This directory contains scripts for spectral validation and experiment analysis.

## Scripts

### `tools/dev/generate_synthetic_traces.py`

Generates synthetic traces for **development and testing purposes only**.

**⚠️ IMPORTANT: Synthetic traces are NOT suitable for evidence validation.**

#### Usage

```bash
# Generate synthetic traces (for development/testing only)
python tools/dev/generate_synthetic_traces.py --n 30 --seed 42 --out-dir dev_artifacts/experiment_traces
```

#### Safety Mechanisms

All synthetic traces are automatically marked with `"data_source": "synthetic"` in the JSON output. This ensures they can be detected and rejected during evidence validation.

### `run_experiment.py`

Runs spectral validation experiments on traces.

#### Usage

```bash
# Run on all scenarios
python analysis/run_experiment.py --all-scenarios --k 10

# Run on specific scenario
python analysis/run_experiment.py --scenario code_backdoor_injection --k 10

# Run on custom traces directory
python analysis/run_experiment.py --traces-dir path/to/traces --output-dir path/to/reports --k 10
```

#### Data Source Detection

The script automatically detects the `data_source` field from input traces and propagates it to the validation reports:
- If any trace has `"data_source": "synthetic"`, the report will be marked as synthetic
- If all traces have `"data_source": "real_traces_only"`, the report will be marked as real
- Legacy traces without the field are treated as unknown

## Workflow Safety

### Development Testing (spectral_validation.yml)

By default, synthetic trace generation is **disabled** (`ALLOW_SYNTHETIC_TRACES: "0"`). 

To enable for development testing, set `ALLOW_SYNTHETIC_TRACES: "1"` in the workflow environment.

### Evidence Validation

Evidence runs should enforce that all traces and reports are marked as real-only:

1. ✅ Verify all validation reports have `"data_source": "real_traces_only"`
2. ✅ Verify all experiment traces have `"data_source": "real_traces_only"`
3. ❌ **Fail the workflow** if any synthetic or unmarked data is detected

This ensures synthetic traces can never accidentally be used as evidence.

## For Real Agent Traces

To generate real agent traces, use:

```bash
python tools/real_agents_hf/run_real_agents.py
```

These traces will be marked with `"data_source": "real_traces_only"` and are suitable for evidence validation.
