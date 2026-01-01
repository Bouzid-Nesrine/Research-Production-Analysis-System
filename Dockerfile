FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy only requirements first (for better caching)
COPY backend/requirements.txt ./backend/
COPY RAG/requirements.txt ./RAG/

# Install Python dependencies with optimization
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt && \
    pip install --no-cache-dir -r RAG/requirements.txt && \
    pip install --no-cache-dir gunicorn && \
    rm -rf /root/.cache/pip

# Copy application code
COPY backend/ ./backend/
COPY RAG/ ./RAG/

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "backend.app:app"]
