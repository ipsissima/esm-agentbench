FROM python:3.11-slim

WORKDIR /app

# Set model cache directory to a location accessible to all users
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HF_HOME=/app/models

COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir sentence-transformers scikit-learn

# Pre-download the sentence-transformers model during build
# This "bakes" the model into the image, bypassing runtime network restrictions
RUN mkdir -p /app/models \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && chmod -R 755 /app/models

COPY . /app

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/demo_traces \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

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

CMD ["python", "-m", "esmassessor.green_server", "--host", "0.0.0.0", "--port", "8080", "--serve-only"]
