FROM python:3.11-slim

WORKDIR /app

# Set model cache directory to a location accessible to all users
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HF_HOME=/app/models

COPY requirements.txt /app/
# Install CPU-only torch first to avoid downloading CUDA libs
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir sentence-transformers scikit-learn

# Pre-download the sentence-transformers model during build
# This avoids network access at runtime and the unshare permission issue
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
