#!/usr/bin/env python
"""Docker healthcheck script for ESM-AgentBench.

This healthcheck verifies that the essential modules are importable
and the environment is correctly set up for judge mode execution.
"""
from __future__ import annotations

import sys

try:
    # Core dependencies
    import numpy
    import scipy
    import sklearn

    # Certificate computation
    from certificates.make_certificate import compute_certificate

    # Real agent infrastructure
    from tools.real_agents_hf.inference import load_model_configs

    # Verify model configs are loadable
    configs = load_model_configs()
    assert len(configs) > 0, "No model configs found"

    # Verify tiny-test model is configured
    model_names = [c.name for c in configs]
    assert "tiny-test" in model_names, "tiny-test model not configured"

    sys.exit(0)
except Exception as e:
    print(f"Healthcheck failed: {e}", file=sys.stderr)
    sys.exit(1)
