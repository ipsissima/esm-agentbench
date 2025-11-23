# Use a slim Python base image suitable for production
FROM python:3.11-slim

# Set application directory
WORKDIR /app

# Copy dependency file separately to leverage Docker layer caching
# (application code changes will not invalidate the dependency layer)
COPY requirements.txt /app/

# Install Python dependencies without caching to keep image small
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Create a non-root user for running the application to improve security
RUN useradd --create-home --shell /bin/bash appuser

# Switch to the non-root user for runtime
USER appuser

# Expose application port
EXPOSE 8080

# Optional healthcheck to detect unresponsive containers quickly
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f -s --max-time 3 http://127.0.0.1:8080/.well-known/agent-card.json || exit 1

# Run the application with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "assessor.app:app", "--workers", "1", "--timeout", "30"]
