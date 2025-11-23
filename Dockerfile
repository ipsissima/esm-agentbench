# Use a slim Python base image suitable for production
FROM python:3.11-slim

# Set application directory
WORKDIR /app

# Copy dependency file first so that dependency installs can be layer-cached even when
# application code changes, reducing rebuild time.
COPY requirements.txt /app/

# Install minimal build tools for wheels that sentence-transformers depends on.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies without caching to keep image small; upgrade pip first
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Pre-download the sentence-transformers model for offline use. To avoid CI rebuilds,
# you can instead pre-populate the cache with:
#   docker build --build-arg TRANSFORMERS_OFFLINE=1 ...
# or run `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`
# in CI and bake the layer. Expect ~90MB extra size from this step.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create a dedicated non-root runtime user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy the rest of the application code
COPY . /app

# Ensure application directories exist and are owned by appuser before dropping
# privileges so the app can write to /app and /app/demo_traces.
RUN mkdir -p /app/demo_traces \
    && chown -R appuser:appuser /app

# Switch to the non-root user for runtime
USER appuser

# Expose application port
EXPOSE 8080

# Healthcheck relies only on Python stdlib (no curl needed) to probe the agent card endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import sys,json,urllib.request; url='http://127.0.0.1:8080/.well-known/agent-card.json'; resp=urllib.request.urlopen(url, timeout=3); json.load(resp); sys.exit(0)"]

# Run the application with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "assessor.app:app", "--workers", "1", "--timeout", "30"]
