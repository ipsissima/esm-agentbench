# Dockerfile (committed, used for Cloud Run)
FROM python:3.11-slim

WORKDIR /app

# Copy dependency file separately to leverage Docker layer caching
COPY requirements.txt /app/

RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Create a non-root user and ensure /app/demo_traces is writable
RUN groupadd -r appuser || true \
    && useradd --system --create-home --gid appuser --shell /bin/bash appuser \
    && mkdir -p /app/demo_traces \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python - <<'PYCODE' || exit 1
import json, sys
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
PYCODE

CMD ["gunicorn", "-b", "0.0.0.0:8080", "assessor.app:app", "--workers", "1", "--timeout", "30"]
