# ─────────────────────────────────────────────────────────────
# Dockerfile
# Data Pipeline Debug Environment
# ─────────────────────────────────────────────────────────────
#
# Runs the FastAPI environment server on port 7860.
# Port 7860 is required for Hugging Face Spaces.
#
# Build:
#   docker build -t data-pipeline-env .
#
# Run:
#   docker run -p 7860:7860 data-pipeline-env
#
# Test:
#   curl -X POST http://localhost:7860/reset \
#        -H "Content-Type: application/json" \
#        -d '{"task": "easy"}'
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata labels
LABEL maintainer="data-pipeline-debug-team"
LABEL description="OpenEnv RL environment for SQL data pipeline debugging"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=7860

# Install dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Expose port 7860 (required for Hugging Face Spaces)
EXPOSE 7860

# Expose port 7860 (Gradio UI) and 8000 (FastAPI Backend)
EXPOSE 7860 8000

# Start the API backend on the mandatory port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
