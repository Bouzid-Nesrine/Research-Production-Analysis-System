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

# Install Python dependencies with aggressive optimization
# Install minimal PyTorch (CPU only, no CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.1.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
        flask==2.3.0 \
        flask-cors==4.0.0 \
        werkzeug==2.3.0 \
        requests==2.31.0 \
        chromadb==0.4.22 \
        sentence-transformers==2.2.2 \
        google-generativeai==0.3.2 \
        python-dotenv==1.0.0 \
        gunicorn==21.2.0 \
        numpy==1.24.3 \
        tqdm==4.65.0 && \
    rm -rf /root/.cache/pip /tmp/*

# Copy application code (excluding large files via .dockerignore)
COPY backend/ ./backend/
COPY RAG/ ./RAG/

# Create necessary directories
RUN mkdir -p backend/uploads RAG/logs RAG/chroma_db

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "backend.app:app"]
