## Demo & Appendix
- Run `./demo.sh` to start the assessor (if needed) and execute the offline SWE demo; results land in `demo_swe/report.json` and `demo_traces/`.
- The math appendix is documented in [DOCS/appendix.md](DOCS/appendix.md), including the conservative theoretical bound.
- CI runs `pytest -q` and optionally builds the Docker image via GitHub Actions; locally, run `pytest -q` before pushing.

# ESM-AgentBench Green Assessor

This repository contains a minimal, judge-friendly "green assessor" for the ESM-AgentBench competition. It exposes a small Flask service that can serve the agent card, run a demonstration episode, save traces, and compute spectral certificates that summarize stability and dynamical properties of agent traces. The project emphasizes deterministic local behavior with optional OpenAI integrations when an `OPENAI_API_KEY` is available.

Competition info deck: [AgentBeats Info Session](file:///mnt/data/agentbeats-competition-info-session-deck.pdf)

## Quickstart
1. Create and activate a virtual environment (Python 3.9+):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the service:
   ```bash
   python assessor/app.py
   ```
   The app listens on port `8080` by default.

## Deployment helper: Cloud Run
If you have gcloud access, you can deploy and auto-update the agent card entrypoint with:

```bash
scripts/update_entrypoint_cloudrun.sh --project <gcp-project> [--region <region>] [--service <name>]
```

The script deploys to Cloud Run from source, retrieves the public HTTPS URL, updates `agent_card.toml` to point at `/.well-known/agent-card.json`, and commits the change. Use `--force` if you need to run with a dirty working tree.

## API
### `GET /.well-known/agent-card.json`
Returns the contents of `agent_card.toml` as JSON. If the file is missing, a 404 JSON error is returned.

### `POST /run_episode`
Request body:
```json
{
  "prompt": "...",
  "tests": [
    {"name": "t1", "script": "assert 2+2 == 4"}
  ]
}
```
Behavior:
- Runs a demo agent against the prompt (OpenAI-backed if configured, deterministic fallback otherwise).
- Executes provided unit tests in a sandboxed temporary directory (`timeout=5`, `shell=False`).
- Writes the trace to `demo_traces/<episode_id>.json`.
- Computes spectral metrics from trace embeddings.

Response:
```json
{
  "episode_id": "...",
  "task_success": true,
  "task_score": 1.0,
  "spectral_metrics": {
    "pca_explained": 0.95,
    "max_eig": 0.7,
    "spectral_gap": 0.2,
    "residual": 0.01,
    "confidence": 0.99
  },
  "trace_path": "demo_traces/<episode_id>.json"
}
```

## Spectral metrics
- **pca_explained**: Total variance captured by the PCA projection (fraction).
- **max_eig**: Largest eigenvalue magnitude of the Koopman proxy; values <1 suggest stability.
- **spectral_gap**: Difference between the largest two eigenvalue magnitudes; larger gaps indicate dominant dynamics.
- **residual**: Normalized reconstruction error of the linear proxy; smaller is better.
- **confidence**: Heuristic `1 - residual`, clipped at 0.

## Development
- Run tests with `pytest`.
- Optional OpenAI usage is gated by the `OPENAI_API_KEY` environment variable; without it, deterministic local embeddings and agent responses are used.
