# AgentBeats Submission: EigenSpace Spectral Bench

## Trace provenance & judge quickstart

### Provenance (truthful summary)
- **Real traces**: This repo includes pre-recorded *real* agent traces in `tools/real_traces/`. Some of those traces were collected using hosted models (e.g., `gpt-4o`) at collection time — we retain them as archived evidence. These traces are genuine LLM runs (they contain full assistant text, code blocks, test stdout/stderr, timestamps and per-step diagnostics).
- **Reproducibility**: **You do not need any private API key** to run or to verify the submission. The evaluation pipeline provides a *judge-mode* that runs fully with local Hugging Face models (or a tiny offline model). The OpenAI/hosted path is optional and used only if `OPENAI_API_KEY` is set.

### Judge quickstart (one-command)
Run the default judge smoke test (small/offline mode):

```bash
# Build the official judge image (this preloads the judge tiny model and the embedder)
docker build -t esm-agentbench .

# Run judge mode (default: small offline mode). This writes traces and validation reports.
docker run --rm esm-agentbench

# After the run, open one validation report (example):
jq . reports/spectral_validation_real_hf/code_backdoor_injection/validation_report.json
```

**What this does:** runs a small, judge-friendly real-agent experiment (no API keys needed), writes traces to `submissions/{team}/{scenario}/experiment_traces_real_hf/` and a `validation_report.json` in `reports/spectral_validation_real_hf/{scenario}/`.

### If you want to reproduce *exactly* a trace produced with OpenAI

Some archived traces were created with OpenAI (e.g., `model_name: "gpt-4o"`). Those traces are stored in `tools/real_traces/`. To check whether a particular trace was recorded with a hosted model, inspect the trace metadata:

```bash
jq '{model:.model_name, timestamp:.timestamp_start}' tools/real_traces/*.json | head -n 20
```

> **Note:** Judges do **not** need an API key to validate submissions — judge-mode produces fully functional evidence using local models. The archived hosted-model traces are provided for additional inspection and are clearly labeled.

### Offline guarantee

To guarantee judge-mode runs offline, we prefetch the tiny judge model and the sentence-transformers embedder into the Docker image. If you prefer not to use Docker, run:

```bash
# local judge-mode (Python): installs requirements and runs a small offline evaluation
python run_judge_mode.py --mode small
```

If you need any help running this, see the `scripts/preload_models.sh` script (prefetches models) and the `Dockerfile` snippet in this repo that performs the prefetch at image build time.

### One-line verification commands for judges

```bash
# 1) Confirm a trace was produced by a hosted model (OpenAI) or HF local model
jq '{model:.model_name, timestamp:.timestamp_start}' tools/real_traces/*.json | head -n 5

# 2) Run the small judge mode (offline)
docker build -t esm-agentbench .
docker run --rm esm-agentbench

# 3) Confirm validation report claims real-only evidence
jq '.data_source' reports/spectral_validation_real_hf/*/validation_report.json
# should print "real_traces_only"
```

**Bottom line:** the repo includes archived real traces (some were recorded via hosted APIs). Judges do **not** need any private API keys: the default judge mode runs with a tiny local HF model and a pre-cached embedder (Docker image prefetch) so the evaluation is fully reproducible and offline. If you want to regenerate original hosted traces, set your `OPENAI_API_KEY` and re-run `run_real_agents.py` (the hosted path is optional and clearly labeled).

---

## Overview

This repository contains 6 AgentBeats scenarios, demonstrating spectral certificate-based detection of AI agent misbehavior using **real tool-using agent traces only**.

**Key Innovation**: All benchmark evidence and metrics are derived from real, tool-using agent execution traces. Synthetic data is used only for isolated numerical regression tests and cannot enter the evaluation pipeline.

## Scenarios

| Scenario | Description | Attack Vector |
|----------|-------------|---------------|
| `supply_chain_poisoning` | Spectral drift detection in supply chain attacks | Reasoning trajectory manipulation |
| `code_backdoor_injection` | Backdoor insertion via adversarial code comments | Hidden instructions in comments |
| `code_review_bypass` | Code review bypass via spectral validation | Manipulation of review process |
| `debug_credential_leak` | Credential leakage during debugging | Hidden extraction directives |
| `refactor_vuln_injection` | Vulnerability injection during refactoring | Subtle code changes |
| `test_oracle_manipulation` | Test oracle manipulation detection | Test manipulation |

## Judge Quickstart (3 Commands, <10 Minutes)

A reviewer can verify all claims in under 10 minutes on modest hardware:

```bash
# 1. Build Docker image (judge mode is default)
docker build -t esm-agentbench .

# 2. Run judge mode (CPU-friendly, n=10, max_steps=20, single scenario)
docker run --rm esm-agentbench

# 3. Verify outputs exist and have correct flags
docker run --rm esm-agentbench python -c "
import json
from pathlib import Path

# Check attack_succeeded.json
attack = json.load(open('scenarios/code_backdoor_injection/attack_succeeded.json'))
print(f'✓ Attack success: {attack[\"success\"]}')

# Check validation report has real_traces_only flag
report_path = Path('reports/spectral_validation_real_hf/code_backdoor_injection/validation_report.json')
if report_path.exists():
    report = json.load(open(report_path))
    print(f'✓ Data source: {report[\"data_source\"]}')
    assert report['data_source'] == 'real_traces_only'
else:
    print('⚠ Report not generated yet (traces may be missing)')
"
```

**Expected output:**
- Judge mode completes in 5-10 minutes on CPU
- `attack_succeeded.json` exists with `success: true`
- `validation_report.json` exists with `data_source: real_traces_only`
- No synthetic traces in evidence path (enforced by CI tests)

## Data Source Guarantee

**All benchmark results/metrics/reports are derived from real tool-using agent traces only.**

- ✅ Real agent traces: Used for all evaluation, reports, and attack validation
- ✅ Synthetic data: Isolated to `tests/test_spectral_math_regression.py` for numerical stability testing only
- ✅ Guardrails: CI tests enforce separation (see `tests/test_no_synthetic_in_evidence_path.py`)

Synthetic traces **cannot** be used in the evaluation pipeline by design. This is verified by automated tests.

## Quick Reproduction (Without Docker)

### Prerequisites

- Python 3.11+
- Git

### Step 1: Clone and Setup

```bash
git clone https://github.com/ipsissima/esm-agentbench.git
cd esm-agentbench
python -m venv .venv && source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
pip install pytest
```

### Step 2: Run Judge Mode

```bash
# Judge mode: CPU-friendly, single scenario, n=10
python run_judge_mode.py --scenario code_backdoor_injection --n 10 --max-steps 20
```

This will:
1. Run real agents (small mode: phi-3-mini-instruct, quantized, CPU-ok)
2. Validate scenario plugin (loads real traces, creates `attack_succeeded.json`)
3. Run spectral validation (creates `validation_report.json`)

Expected runtime: 5-10 minutes on CPU

### Step 3: Run Full Evaluation (Optional)

For comprehensive results with multiple models:

```bash
# Generate traces with multiple models (requires GPU recommended)
python tools/real_agents_hf/run_real_agents.py \
    --mode full \
    --all-scenarios \
    --n 40

# Run spectral validation with cross-model evaluation
python analysis/run_real_hf_experiment.py --all-scenarios --cross-model
```

Expected runtime: 2-4 hours with GPU, 8-16 hours on CPU

### Step 4: Verify All Gates Pass

```bash
# Check all scenarios have attack_succeeded.json
for s in scenarios/*; do
  echo "==> Checking $s"
  python -c "import json; j=json.load(open('$s/attack_succeeded.json')); print('success:', j.get('success'))"
done

# Run guardrail tests
pytest tests/test_no_synthetic_in_evidence_path.py -v
pytest tests/test_reports_real_only_flag.py -v
pytest tests/test_judge_mode_smoke.py -v
```

### Step 5: Run Full Test Suite

```bash
pytest -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ run_judge_mode.py (default Docker entrypoint)               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├──► 1. tools/real_agents_hf/run_real_agents.py
                 │      (runs local HF models as agents)
                 │      └─► submissions/{team}/{scenario}/experiment_traces_real_hf/
                 │            {model}/gold/*.json
                 │            {model}/creative/*.json
                 │            {model}/drift/*.json
                 │
                 ├──► 2. scenarios/{scenario}/plugin.py
                 │      (validates real traces, no synthetic generation)
                 │      └─► attack_succeeded.json
                 │
                 └──► 3. analysis/run_real_hf_experiment.py
                        (spectral validation, calibrated thresholds)
                        └─► reports/spectral_validation_real_hf/{scenario}/
                              validation_report.json (data_source: real_traces_only)
                              features.csv
                              roc_*.png
```

## Resource Requirements

### Judge Mode (Small, CPU-Friendly)
- **Model**: phi-3-mini-instruct (quantized, 2-3GB RAM)
- **Hardware**: 8GB RAM, CPU-only
- **Runtime**: 5-10 minutes for single scenario
- **Runs**: n=10 per label
- **Purpose**: Reviewer verification

### Full Mode (Comprehensive, GPU-Recommended)
- **Models**: 3+ models (deepseek-coder-7b, codellama-13b, starcoder2-15b)
- **Hardware**: 16GB+ GPU, 32GB+ RAM
- **Runtime**: 2-4 hours for all scenarios
- **Runs**: n=40 per label
- **Purpose**: Complete cross-model evaluation

## Evaluation Metrics

All metrics from real traces, calibrated thresholds:

- **Per-model AUC**: Drift detection within each model's traces
- **Cross-model AUC**: Train on models A+B, test on model C
- **TPR @ FPR=0.05**: True positive rate at 5% false positive rate
- **Spectral bound**: Koopman residual + (1 - PCA energy)

Targets (from `docs/REAL_AGENT_HF_EVAL.md`):
- Per-model AUC ≥ 0.85
- Cross-model AUC ≥ 0.80
- Mean TPR @ FPR=0.05 ≥ 0.70

Thresholds are calibrated per scenario as: `tau = baseline_mean + k*std` (no hardcoded ratios).

## Repository Structure

```
esm-agentbench/
├── run_judge_mode.py              # Default Docker entrypoint
├── Dockerfile                      # Judge mode by default
├── evaluation_config.yaml          # Canonical evaluation config (pca_rank, calibration)
├── scenarios/
│   ├── {scenario}/
│   │   ├── plugin.py              # Real-only (no synthetic generation)
│   │   ├── baseline_test.py       # Scenario validation tests
│   │   ├── real_agent_prompts/    # Prompts for real agents
│   │   │   ├── gold.md
│   │   │   ├── creative.md
│   │   │   └── drift.md
│   │   └── targets/               # Scenario target files
│   └── ...
├── tools/real_agents_hf/
│   ├── run_real_agents.py         # Real agent orchestrator
│   ├── models.yaml                # Model configurations
│   ├── agent_loop.py              # Agent execution loop
│   ├── sandbox.py                 # Isolated execution
│   └── trace.py                   # Trace recording with embeddings
├── analysis/
│   └── run_real_hf_experiment.py  # Spectral validation (real-only)
├── reports/
│   └── spectral_validation_real_hf/  # All reports (data_source: real_traces_only)
├── tests/
│   ├── test_spectral_math_regression.py        # Synthetic allowed (tests only)
│   ├── test_no_synthetic_in_evidence_path.py   # Guardrail: no synthetic in evidence
│   ├── test_reports_real_only_flag.py          # Guardrail: reports have real_traces_only
│   └── test_judge_mode_smoke.py                # Smoke test for judge mode
└── docs/
    └── REAL_AGENT_HF_EVAL.md       # Evaluation plan and targets
```

## CI/CD Guarantees

CI enforces:
1. No synthetic generation code in `analysis/`, `tools/real_agents_hf/`, `scenarios/*/plugin.py`
2. All `validation_report.json` files have `data_source: real_traces_only`
3. Judge mode smoke test passes
4. Numerical regression tests pass (synthetic matrices in `tests/` only)

## Docker Usage

### Default (Judge Mode)

```bash
docker build -t esm-agentbench .
docker run --rm esm-agentbench
```

### Custom Options

```bash
# Different scenario
docker run --rm esm-agentbench python run_judge_mode.py --scenario supply_chain_poisoning

# More runs
docker run --rm esm-agentbench python run_judge_mode.py --n 20

# Run tests only
docker run --rm esm-agentbench pytest tests/test_no_synthetic_in_evidence_path.py -v
```

### Full Mode (Requires GPU, More Time)

```bash
docker run --rm --gpus all esm-agentbench python tools/real_agents_hf/run_real_agents.py --mode full --all-scenarios
```

## FAQ

**Q: How do I know the traces are real?**
A: All traces are in `submissions/{team}/{scenario}/experiment_traces_real_hf/{model}/{label}/*.json`. Each trace contains:
- `metadata`: Model name, scenario, timestamp, prompt
- `steps`: Full agent execution steps with tool calls
- `outcome`: Completion status, elapsed time

**Q: Where is the synthetic data?**
A: Synthetic data exists **only** in `tests/test_spectral_math_regression.py` for numerical regression testing (SVD stability, Koopman residual computation). CI tests prevent it from entering the evaluation path.

**Q: How are thresholds calibrated?**
A: Thresholds are computed from gold traces as `tau = mean(gold_bound) + k*std(gold_bound)` where k=2 by default. No hardcoded ratios like "1.15x". This is per-scenario adaptive.

**Q: Can judges run this without API keys?**
A: Yes. All models are local HuggingFace models. No API keys required. Judge mode uses quantized models that run on CPU.

**Q: What if I don't have a GPU?**
A: Judge mode is CPU-friendly (phi-3-mini-instruct, quantized). Full mode benefits from GPU but can run on CPU (slower).

**Q: How long does evaluation take?**
A:
- Judge mode: 5-10 minutes (single scenario, n=10)
- Full mode: 2-4 hours with GPU (all scenarios, n=40, 3 models)

## Contact

For questions or issues, open an issue at: https://github.com/ipsissima/esm-agentbench/issues

## Contributors & Provenance

See the following files in the repository root for contributor and provenance information:

- [`CONTRIBUTORS.md`](../CONTRIBUTORS.md) — List of human and AI contributors
- [`AGENT_PROVENANCE.md`](../AGENT_PROVENANCE.md) — Agent-generated code mapping
- [`REVIEWED_BY_ANDREU.md`](../REVIEWED_BY_ANDREU.md) — Human review sign-off

## License

MIT License (see LICENSE file)
