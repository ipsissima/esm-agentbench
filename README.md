ESM-AgentBench
ESM-AgentBench — Spectral / Eigenspace Certificates for Agent Evaluation

This repo contains the minimal green-assessor and tooling for Phase-1 of the AgentX–AgentBeats competition:

Port of a SWE-bench Lite subset into AgentBeats as a green assessor (assessor/).

A certificate pipeline that computes a finite-rank Koopman proxy over agent interaction traces and emits certified spectral metrics:

max_eig (maximum eigenvalue magnitude),

spectral_gap (separation between leading eigenvalues),

residual (least-squares reconstruction error).

Demo traces and scripts to reproduce the certificate outputs.

Homepage
/mnt/data/agentbeats-competition-info-session-deck.pdf

Quickstart (development)

bash
Copia el codi
git clone https://github.com/<owner>/esm-agentbench.git
cd esm-agentbench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# run a demo local episode
python assessor/kickoff.py
# run the Flask assessor server
python assessor/app.py
Interpretation for judges

The spectral metrics measure the low-rank, linearisable dynamics of an agent’s internal trajectory (messages, tool calls, code edits) embedded into vector form. The metrics are accompanied by a residual error term for outcome validity and reproducibility.

The implementation reuses the EigenSpace certificate pipeline (experimental research code) for rigorous certificate extraction.

License & IP

MIT License. (If you require different IP handling due to patent concerns, see ULELAT/ or contact the author.)

Contact

Andreu Ballús Santacana — andreuballus@gmail.com

javascript
Copia el codi
