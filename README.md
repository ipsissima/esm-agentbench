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
