FROM python:3.11-slim

WORKDIR /app

# Set model cache directory to a location accessible to all users
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HF_HOME=/app/models

COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir sentence-transformers scikit-learn

# Create model directory
RUN mkdir -p ${SENTENCE_TRANSFORMERS_HOME}

# Bake the model into the image (runs as root during build, has network access in CI)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . /app

# Create user and set permissions (including model directory)
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/demo_traces \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python /app/healthcheck.py

CMD ["python", "-m", "esmassessor.green_server", "--host", "0.0.0.0", "--port", "8080", "--serve-only"]
