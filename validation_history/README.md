# Validation History

This directory contains historical validation results from production benchmark runs against real AI traces.

## Current Validation (20251214_020930)

### Dataset Overview
- **Total traces analyzed:** 200 real AI-generated traces
- **Gold runs:** 60 (GPT-4o, temperature=0.0)
- **Creative runs:** 40 (GPT-4o with unconventional approaches)
- **Drift runs:** 60 (GPT-3.5-turbo, temperature=1.3)
- **Poison runs:** 40 (GPT-4o with adversarial prompts)

### Validation Artifacts

#### Timestamped Files
Files are named with format: `YYYYMMDD_HHMMSS_<git-hash>_<type>.<ext>`

- `*_validation.json` - Full validation results with per-trace theoretical bounds
- `*_harvest.csv` - Statistical summary by dataset tag (gold/creative/drift/poison)
- `*_harvest.png` - Violin plot visualization of theoretical bounds
- `*_summary.csv` - Detailed harvest summary with all trace metadata

#### Symlinks to Latest
- `latest_validation.json` → most recent validation results
- `latest_harvest.csv` → most recent harvest statistics
- `latest_harvest.png` → most recent visualization
- `latest_summary.csv` → most recent harvest summary

## Validation Methodology

### Trace Generation
Real traces are generated using `tools/harvest_data.py`:
```bash
python tools/harvest_data.py \
  --episodes demo_swe/episodes \
  --outdir tools/real_traces_harvested \
  --gold_runs 15 --creative_runs 10 \
  --drift_runs 15 --poison_runs 10 \
  --concurrency 5
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

The current validation uses TF-IDF embeddings (fallback mode). For production-grade discrimination between creativity and drift, semantic embeddings are recommended:
- OpenAI embeddings (requires `OPENAI_API_KEY`)
- sentence-transformers (requires network access to huggingface.co)

TF-IDF measures vocabulary difference rather than semantic coherence, which may not reliably distinguish creative solutions from drift.

## Historical Context

This validation replaces earlier synthetic "toy data" with real AI traces, demonstrating the benchmark's functionality on production LLM outputs from GPT-4o and GPT-3.5-turbo.
