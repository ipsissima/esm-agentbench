# Validation History

This directory contains historical validation results from production benchmark runs against real AI traces.

## IMPORTANT: Known Issue with Deterministic Fallback

**Issue discovered 2024-12-14:** Previous validation runs (20251214_*) were affected by silent API fallback to the `deterministic_agent`, which produces identical trace content regardless of the intended category (creative, drift, poison, etc.).

### Root Cause
When the OpenAI API is unavailable (bad key, network issues), the `call_agent` function in `assessor/kickoff.py` silently falls back to `deterministic_agent`, which produces hardcoded responses like:
- "Step 1: Define Fibonacci base cases n=0->0 and n=1->1."
- "Step 2: Iterate from 2..n accumulating previous two numbers."

This causes all traces to contain identical content, making the validation meaningless.

### Detection
Corrupted traces can be identified by checking if the first CoT step contains deterministic signatures:
```python
DETERMINISTIC_SIGNATURES = [
    "Define Fibonacci base cases",
    "Iterate from 2..n accumulating",
    "Maintain (a,b) and update",
]
```

### Prevention
Use the `--require-api` and `--verify-traces` flags when harvesting:
```bash
python tools/harvest_data.py \
  --episodes demo_swe/episodes \
  --outdir tools/real_traces \
  --require-api \
  --verify-traces \
  --gold_runs 15 --creative_runs 10 \
  --drift_runs 15 --poison_runs 10
```

## Validation Artifacts

### Timestamped Files
Files are named with format: `YYYYMMDD_HHMMSS_<git-hash>_<type>.<ext>`

- `*_validation.json` - Full validation results with per-trace theoretical bounds
- `*_harvest.csv` - Statistical summary by dataset tag (gold/creative/drift/poison)
- `*_harvest.png` - Violin plot visualization of theoretical bounds
- `*_summary.csv` - Detailed harvest summary with all trace metadata

### Symlinks to Latest
- `latest_validation.json` -> most recent validation results
- `latest_harvest.csv` -> most recent harvest statistics
- `latest_harvest.png` -> most recent visualization
- `latest_summary.csv` -> most recent harvest summary

## Validation Methodology

### Trace Generation
Real traces are generated using `tools/harvest_data.py`:
```bash
python tools/harvest_data.py \
  --episodes demo_swe/episodes \
  --outdir tools/real_traces_harvested \
  --gold_runs 15 --creative_runs 10 \
  --drift_runs 15 --poison_runs 10 \
  --concurrency 5 \
  --require-api \
  --verify-traces
```

### Validation Process
Traces are validated using `tools/validate_real_traces.py`:
```bash
python tools/validate_real_traces.py
```

This computes spectral certificates for all traces and tests the metric's ability to discriminate between:
1. **Creative (correct but unconventional)** vs **Drift (incorrect/hallucination)**
2. **All good traces** vs **All bad traces**

### Key Metrics
- **theoretical_bound** - Core spectral certificate metric (lower = more coherent)
- **residual** - Koopman approximation error
- **pca_explained** - Variance captured by principal components

## Embedding Method Note

Validation requires **semantic embeddings** (OpenAI or sentence-transformers) for reliable results. TF-IDF fallback is NOT suitable for validation as it measures vocabulary difference, not semantic coherence.

### Valid Embedding Methods
- `openai` - Uses text-embedding-3-small (recommended)
- `sentence-transformers` - Uses all-MiniLM-L6-v2 (requires network on first use)

### Invalid Embedding Method
- `tfidf` - Will produce FALSE NEGATIVES for creative solutions that use different vocabulary

## Historical Context

Corrupted validation files have been removed from this directory. Valid traces remain in `tools/real_traces/` (150 traces after cleanup).
