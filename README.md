## Coherence & Early Warning (Executive summary)
We diagnose **chain-of-thought coherence** using Koopman-style residuals instead of vague stability claims.
The tool is a **heuristic diagnostic** validated empirically on cached real traces—treat it as an early-warning light, not a proof of safety.
Run cached demos safely offline: `./demo.sh` (or Docker) executes good/bad traces, renders `demo_swe/fig_hallucination.png`, and writes `demo_traces/*.json` plus `demo_swe/report.json`.
Replay cached traces without API keys via `python -m assessor.kickoff` options `trace_path` / `use_cached_real`, and calibrate thresholds with `python tools/calibrate_threshold.py`.
CI can disable networking for tests (`--network none`, `DOCKER_IS_AVAILABLE=1`) while local dev can skip risky cases with `SKIP_UNSAFE_TESTS=1`.
The Semantic Oracle nano-judge shadows each reasoning step to flag destructive commands, hallucinated imports, or incoherent pivots. It defaults to `gpt-4o-mini` (override with `SEMANTIC_ORACLE_MODEL`) and can be throttled via `oracle_every_n_steps`, gracefully falling back to offline warnings when no key is present.

## Coherence & Early Warning
Agents that think well move smoothly through semantic space; jagged jumps often signal hallucinations or incoherent pivots. This repo now emphasizes *coherence* over generic “stability,” pairing chain-of-thought traces with Koopman-style drift metrics and an early-warning flag when residuals spike.

- Run `./demo.sh` to launch the assessor (or Docker entrypoint), execute offline SWE episodes, and generate Good vs Bad CoT runs. The hallucination plot saves to `demo_swe/fig_hallucination.png` and the episode report to `demo_swe/report.json` alongside trace JSON in `demo_traces/`.
- Interpret the curves as a **heuristic**: smooth, low cumulative residuals suggest coherent reasoning; sharp jumps (and the `early_warning_step`) highlight semantic drift. Thresholds are tunable and not a proof of safety—treat them as an interpretable early-warning light.
- For reproducible embeddings and offline portability, the Docker image preloads `all-MiniLM-L6-v2`; OpenAI embeddings are used only when an API key is present.

# ESM-AgentBench Green Assessor

This repository contains a minimal, judge-friendly "green assessor" for the ESM-AgentBench competition. It exposes a small Flask service that can serve the agent card, run a demonstration episode, save traces, and compute spectral certificates that summarize coherence and dynamical properties of agent traces. The project emphasizes deterministic local behavior with optional OpenAI integrations when an `OPENAI_API_KEY` is available.

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
3. Run tests:
   ```bash
   pytest -q
   ```
4. Start the demo server locally:
   ```bash
   python -m assessor.app
   ```

## Running the SWE demo
A sample SWE demo can be invoked with:
```bash
./demo.sh
```
This saves a trace JSON to `demo_traces/`, a summary report to `demo_swe/report.json`, and a hallucination coherence figure to `demo_swe/fig_hallucination.png`.

## Certificate computation
The `certificates` module provides Koopman-based certificates over the state space of an MDP using the approach described in “PCA reduction of Koopman operators.” `compute_certificate` computes finite approximations of the Koopman operator which yield certificates describing global long-term properties of dynamical systems. See the mathematical appendix in [DOCS/appendix.md](DOCS/appendix.md) for additional details.

## Configuration
- `OPENAI_API_KEY`: If set, the demo attempts to use OpenAI for completions and embeddings.
- Without an API key, the demo uses a deterministic agent and embedding stack suitable for offline evaluation.

## Harvesting real traces
- Dry-run the harvester without API calls: `python tools/harvest_data.py --episodes demo_swe/episodes --outdir demo_traces --gold_runs 1 --drift_runs 1 --poison_runs 1 --starve_runs 1 --dry-run`
- Full run with budget guardrails: `python tools/harvest_data.py --episodes demo_swe/episodes --outdir demo_traces --gold_runs 5 --drift_runs 5 --poison_runs 5 --starve_runs 5 --api_delay 0.2 --max_api_calls 100 --est_cost_limit 5`

## Deploy to Cloud Run and make agent-card discoverable

### Option A — automated deploy via GitHub Actions (recommended)
1. Add repository secrets:
   - `GCP_PROJECT_ID`: your Google Cloud project id
   - `GCP_SA_KEY`: base64 or raw JSON of a service account key with Cloud Run / Cloud Build permissions
2. In Actions → Workflows → "Deploy to Cloud Run and update agent_card" click **Run workflow**.
3. The workflow will build, deploy to Cloud Run, update `agent_card.toml` and `.well-known/agent-card.json`, and open a PR that updates the repo with the deployed URL.

### Option B — local testing
1. Run the assessor locally: `./demo.sh` (this starts the app at `http://localhost:8080`).
2. Make the agent card point to your local server:
   ```bash
   scripts/make_local_agent_card.sh
   ```
   This sets agent_card.toml entrypoint to http://localhost:8080/.well-known/agent-card.json and writes .well-known/agent-card.json.
3. Confirm the agent card is served:

```bash
curl -sS http://localhost:8080/.well-known/agent-card.json | jq .
```

Notes:

The GitHub Action writes an idempotent .well-known/agent-card.json and opens a PR; maintainer review is recommended before merging.

The workflow uses gcloud and gcloud builds to produce a reproducible container.
