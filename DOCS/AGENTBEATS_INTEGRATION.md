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
- Generate real agent traces using local HuggingFace models:
  ```bash
  python tools/real_agents_hf/run_real_agents.py --scenario code_backdoor_injection --n 20
  ```
- Calibrate thresholds from real traces (dry-run for CI):
  ```bash
  python tools/calibrate_thresholds.py --backend sentence-transformers --trials 4 --dry-run
  ```
- The script writes `certificates/calibration_sentence-transformers.json` and updates `evaluation_config.yaml` with `residual_threshold` and `jump_factor`.

**Note:** All calibration must use real agent traces. Synthetic trace generation is deprecated and isolated in `legacy/` (see `legacy/README.md`).

## Validate Phase-1 scenarios locally

This repository includes 6 Phase-1 scenarios for AgentBeats submission:

1. **code_backdoor_injection**: Backdoor insertion via adversarial code comments
2. **supply_chain_poisoning**: Spectral drift detection in supply chain attacks
3. **test_oracle_manipulation**: Test oracle manipulation detection
4. **code_review_bypass**: Code review bypass via spectral validation
5. **debug_credential_leak**: Credential leakage during debugging
6. **refactor_vuln_injection**: Vulnerability injection during refactoring

### Local Validation Steps

1. Start the green assessor:
   ```bash
   python -m esmassessor.green_server --port 8080 --show-logs
   ```

2. For each scenario, run baseline and plugin:
   ```bash
   cd scenarios/code_backdoor_injection
   python baseline_test.py
   python plugin.py   # writes attack_succeeded.json

   cd ../supply_chain_poisoning
   python baseline_test.py
   python plugin.py

   cd ../test_oracle_manipulation
   python baseline_test.py
   python plugin.py

   # ... repeat for remaining scenarios
   ```

3. Verify all `attack_succeeded.json` files show `"success": true`:
   ```bash
   for s in scenarios/*; do
     echo "==> $s"
     python -c "import json; j=json.load(open('$s/attack_succeeded.json')); print('success:', j.get('success'))"
   done
   ```

4. Run the full validation test suite:
   ```bash
   pytest tests/test_phase1_submission.py -v
   ```

5. (Optional) Run the demo validation:
   ```bash
   python tools/run_demo.py
   ```

### Gate Artifacts

Each scenario produces an `attack_succeeded.json` at its root containing:
- `success`: boolean indicating attack demonstration success
- `evidence`: object with human-readable snippet and trace path
- `successful_rounds`: list of successful attack rounds

Judges will verify that all 6 scenarios have `"success": true`.

### Agent Card

The `agent_card.toml` provides local-run instructions:
```
entrypoint = "LOCAL: start with `python -m esmassessor.green_server --port 8080 --show-logs` then point harness to http://localhost:8080"
```

See also: `SUBMISSIONS/README.md` for a complete reproduction guide.
