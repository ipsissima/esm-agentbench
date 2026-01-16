#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir "huggingface-hub==0.16.4" "sentence-transformers==2.3.1" "transformers>=4.36.0"
python -m pip install --no-cache-dir -r requirements.txt --no-deps
python -m pip install -e .
echo "Dev bootstrap complete. Activate with: . .venv/bin/activate"
