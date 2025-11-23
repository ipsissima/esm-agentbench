"""Project-level sitecustomize to expose demo_swe refactor helpers during tests."""
import os
import sys

REFACTOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "demo_swe", "refactor_sources"))
if REFACTOR_PATH not in sys.path:
    sys.path.insert(0, REFACTOR_PATH)
