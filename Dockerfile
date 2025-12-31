FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY backend/ ./backend/
COPY RAG/ ./RAG/
COPY start_system.sh ./
COPY stop_system.sh ./

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install --no-cache-dir -r RAG/requirements.txt

# Expose Flask port
EXPOSE 5000

# Make scripts executable
RUN chmod +x start_system.sh stop_system.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start the system
CMD ["python3", "backend/app.py"]
