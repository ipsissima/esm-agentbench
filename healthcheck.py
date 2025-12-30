#!/usr/bin/env python
"""Docker healthcheck script for ESM Assessor."""
from __future__ import annotations

import json
import logging
import os
import sys
from urllib.error import URLError
from urllib.request import urlopen

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Allow configuring host and port via environment variables
ASSESSOR_HOST = os.environ.get("ASSESSOR_HOST", "127.0.0.1")
ASSESSOR_PORT = os.environ.get("ASSESSOR_PORT", "8080")
HEALTHCHECK_URL = f"http://{ASSESSOR_HOST}:{ASSESSOR_PORT}/.well-known/agent-card.json"

try:
    with urlopen(HEALTHCHECK_URL, timeout=3) as resp:
        if resp.status != 200:
            logger.error(f"Healthcheck failed: HTTP {resp.status}")
            sys.exit(1)
        json.loads(resp.read())
except (URLError, TimeoutError, json.JSONDecodeError) as e:
    logger.error(f"Healthcheck failed: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error during healthcheck: {e}")
    sys.exit(1)
else:
    sys.exit(0)
