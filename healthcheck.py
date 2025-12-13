#!/usr/bin/env python
"""Docker healthcheck script for ESM Assessor."""
import json
import sys
from urllib.request import urlopen

try:
    with urlopen("http://127.0.0.1:8080/.well-known/agent-card.json", timeout=3) as resp:
        if resp.status != 200:
            sys.exit(1)
        json.loads(resp.read())
except Exception:
    sys.exit(1)
else:
    sys.exit(0)
