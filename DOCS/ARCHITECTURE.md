# System Architecture

- **`assessor/`**: Core episode runner (`kickoff.py`) and embedding logic. Handles trace generation and step execution.
- **`esmassessor/`**: The AgentBeats-compatible Green Agent server. This is the main entry point (`green_server.py`) wrapping the execution logic.
- **`certificates/`**: Mathematical core. Computes spectral certificates, Koopman residuals, and handles the UELAT bridge.
- **`demo_swe/`**: The customized SWE-bench Lite harness. Contains the coding episodes (`episodes/`) and reference implementations.
- **`tools/`**: CLI utilities for running demos (`run_demo.py`), validating metrics, and calibrating thresholds.
