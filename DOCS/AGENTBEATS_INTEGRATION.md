# AgentBeats Integration

## Quickstart
1. Install dependencies (Python 3.11):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the green assessor server:
   ```bash
   python -m esmassessor.green_server --port 8080 --show-logs
   ```
3. Execute the demo episodes and collect certificates:
   ```bash
   python tools/run_demo.py
   ```

## Certificates
Each assessment writes `demo_traces/<assessment>_<participant>_certificate.json` plus an aggregated `demo_swe/report.json`. The `spectral_metrics` block captures:
- `pca_explained`, `tail_energy`: SVD variance coverage and truncation energy loss.
- `sigma_max`, `singular_gap`: Leading singular value and gap (Wedin stability margin).
- `residual`, `theoretical_bound`: Trajectory prediction residual and SVD-based stability bound.
- `task_score`: heuristic participant score; `trace_path`: local trace file.

## Canonical settings
- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (falls back to TF-IDF with 512 features).
- PCA rank: 10 (see `evaluation_config.yaml`).
- Ports: green server `8080` by default.

## Calibration workflow
- Generate or reuse seed traces if desired: `python tools/generate_seed_traces.py --trials 2 --backends sentence-transformers tfidf`.
- Calibrate thresholds (dry-run for CI):
  ```bash
  python tools/calibrate_thresholds.py --backend sentence-transformers --trials 4 --dry-run
  ```
- The script writes `certificates/calibration_sentence-transformers.json` and updates `evaluation_config.yaml` with `residual_threshold` and `jump_factor`.
