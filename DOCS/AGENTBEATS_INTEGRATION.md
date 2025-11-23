# AgentBeats Integration

## Quickstart
1. Install dependencies (Python 3.11):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Import the tutorial assets (optional when archive is available):
   ```bash
   python tools/import_tutorial.py
   ```
3. Run the green assessor server:
   ```bash
   python -m esmassessor.green_server --port 8080 --show-logs
   ```
4. Launch the purple debater in another shell:
   ```bash
   python third_party/tutorial/scenarios/debate/debater.py --host 127.0.0.1 --port 9019 --show-logs
   ```
5. Execute the demo scenario and collect certificates:
   ```bash
   ./agentbeats-run scenarios/esm/scenario.toml --show-logs
   ```

## Certificates
Each assessment writes `demo_traces/<assessment>_<participant>_certificate.json` plus an aggregated `demo_traces/certificate.json`. The `spectral_metrics` block captures:
- `pca_explained`, `pca_tail_estimate`: PCA variance coverage and tail estimate.
- `max_eig`, `spectral_gap`: Koopman eigenvalue magnitude and gap.
- `residual`, `theoretical_bound`: Koopman residual and conservative bound.
- `task_score`: heuristic participant score; `trace_path`: local trace file.

## Canonical settings
- Embeddings: sentence-transformers `all-MiniLM-L6-v2` (falls back to TF-IDF with 512 features).
- PCA rank: 10 (see `evaluation_config.yaml`).
- Ports: green server `8080`, purple `9019` by default.

## Calibration workflow
- Generate or reuse seed traces if desired: `python tools/generate_seed_traces.py --trials 2 --backends sentence-transformers tfidf`.
- Calibrate thresholds (dry-run for CI):
  ```bash
  python tools/calibrate_thresholds.py --backend sentence-transformers --trials 4 --dry-run
  ```
- The script writes `certificates/calibration_sentence-transformers.json` and updates `evaluation_config.yaml` with `residual_threshold` and `jump_factor`.
