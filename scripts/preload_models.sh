#!/usr/bin/env bash
# scripts/preload_models.sh
# Prefetches the embedding model and tiny judge model for offline operation.
# Run this during docker build or before running judge mode.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "Prefetching embedding model and tiny judge model..."

# Python helper to prefetch models (reads models.yaml for tiny-test hf_id)
${PYTHON} - <<'PY'
import yaml, sys
from pathlib import Path

cfg_path = Path("tools/real_agents_hf/models.yaml")
if not cfg_path.exists():
    print("models.yaml not found at", cfg_path)
    sys.exit(1)

cfg = yaml.safe_load(cfg_path.read_text())
models = cfg.get("models", [])
tiny_hf = None
for m in models:
    if m.get("name") == "tiny-test":
        tiny_hf = m.get("hf_id")
        break

embedder = "sentence-transformers/all-MiniLM-L6-v2"

print("Embedder:", embedder)
try:
    from sentence_transformers import SentenceTransformer
    print("Loading embedder...")
    SentenceTransformer(embedder)
    print("Embedder cached.")
except Exception as e:
    print("Failed to preload embedder:", e)
    print("You may need compatible torch and sentence-transformers installed.")

if tiny_hf:
    print("Tiny HF model:", tiny_hf)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("Prefetching tiny HF model...")
        AutoTokenizer.from_pretrained(tiny_hf)
        AutoModelForCausalLM.from_pretrained(tiny_hf)
        print("Tiny model cached.")
    except Exception as e:
        print("Failed to preload tiny model:", e)
        print("If your environment lacks enough RAM, select a smaller model in models.yaml or build the image on a machine with more resources.")
else:
    print("tiny-test model not found in models.yaml. If you rely on tiny offline mode, add tiny-test entry.")
PY

echo "Prefetch complete."
