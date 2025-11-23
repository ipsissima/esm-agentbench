# Canonical evaluation & calibration

This document explains how to reproduce canonical embeddings, generate precomputed seed traces, and calibrate residual thresholds and jump factors for the ESM spectral certificate.

Canonical defaults:
- embedding backend: sentence-transformers / model `all-MiniLM-L6-v2` (384 dims)
- tfidf: ngram_range=(1,2), max_features=512
- OpenAI: `text-embedding-3-small` (if OPENAI_API_KEY present)
- PCA rank: 10
- pivot_spike_factor: 5.0
- post_stabilize_factor: 0.5

Quick commands:
1. Install deps:
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn sentence-transformers PyYAML

2. Generate seed traces and embeddings:
python tools/generate_seed_traces.py --trials 10

3. Calibrate thresholds (30 trials default):
python tools/calibrate_thresholds.py --trials 30

After calibration, `evaluation_config.yaml` will contain `calibration.residual_threshold` and `calibration.jump_factor` set for the canonical backend. Per-backend calibration artifacts are stored under `certificates/calibration_{backend}.json`.
