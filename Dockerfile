# Use a slim Python base image suitable for production
FROM python:3.11-slim

# Set application directory
WORKDIR /app

# Copy dependency file separately to leverage Docker layer caching so code changes
# do not invalidate the dependency install layer unnecessarily
COPY requirements.txt /app/

# Install Python dependencies without caching to keep image small; upgrade pip first
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Create a non-root user for running the application to improve security. Ensure
# application directories exist and are owned by the runtime user before
# dropping privileges so the app can write to /app and /app/demo_traces.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/demo_traces \
    && chown -R appuser:appuser /app

# Switch to the non-root user for runtime
USER appuser

# Expose application port
EXPOSE 8080

# Healthcheck relies only on Python stdlib (no curl in slim image) to validate
# the agent card endpoint quickly.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python - <<'PYCODE' || exit 1
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
PYCODE

# Run the application with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "assessor.app:app", "--workers", "1", "--timeout", "30"]
