"""Project-level sitecustomize to expose demo paths during tests."""
import os
import sys
from pathlib import Path

REFACTOR_PATH = Path(__file__).resolve().parent / "demo_swe" / "refactor_sources"
if str(REFACTOR_PATH) not in sys.path:
    sys.path.insert(0, str(REFACTOR_PATH))
