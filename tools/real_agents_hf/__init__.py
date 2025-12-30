"""Real agent evaluation using local Hugging Face models.

This package provides infrastructure for running multiple local coding agents
through Phase-1 scenarios and collecting non-synthetic traces for spectral
validation.

Main components:
- inference: Local model backends (transformers, vLLM)
- sandbox: Safe execution environment
- tools: Agent tools (file ops, commands, etc.)
- agent_loop: Agent execution with tool protocol
- embeddings: Local embedding generation
- trace: Trace recording and storage
- run_real_agents: Multi-model runner CLI

All operations are local - no external API calls required.
"""

__version__ = "0.1.0"
