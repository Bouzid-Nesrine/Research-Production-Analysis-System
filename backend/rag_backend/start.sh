#!/bin/bash

# Research Paper Classification Backend Startup Script
# This script starts all required services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# GROBID Configuration
GROBID_IMAGE="lfoppiano/grobid:0.8.1"

echo "========================================"
echo "Research Paper Classification Backend"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Docker
echo "Checking Docker..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker is installed${NC}"
else
    echo -e "${RED}✗ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker daemon is running
if docker info &> /dev/null; then
    echo -e "${GREEN}✓ Docker daemon is running${NC}"
else
    echo -e "${RED}✗ Docker daemon is not running. Please start Docker.${NC}"
    exit 1
fi

# Check GROBID container
echo ""
echo "Checking GROBID container..."
if docker ps --format '{{.Names}}' | grep -q "^grobid$"; then
    echo -e "${GREEN}✓ GROBID container is running${NC}"
else
    echo -e "${YELLOW}Starting GROBID container...${NC}"
    
    # Check if container exists but is stopped
    if docker ps -a --format '{{.Names}}' | grep -q "^grobid$"; then
        # Remove old container to ensure fresh start with correct flags
        echo "Removing old GROBID container..."
        docker rm -f grobid > /dev/null 2>&1
    fi
    
    # Create and run container with cgroups v2 compatibility
    echo "Creating and starting GROBID container..."
    docker run -d \
        --name grobid \
        -p 8070:8070 \
        -e JAVA_OPTS="-XX:-UseContainerSupport" \
        $GROBID_IMAGE
    
    echo "Waiting for GROBID to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:8070/api/isalive > /dev/null 2>&1; then
            echo -e "${GREEN}✓ GROBID is ready${NC}"
            break
        fi
        sleep 1
        if [ $i -eq 60 ]; then
            echo -e "${RED}✗ GROBID failed to start within 60 seconds${NC}"
            exit 1
        fi
    done
fi

# Activate Python environment (if using virtual env)
echo ""
echo "Setting up Python environment..."

cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "Activating project virtual environment..."
    source ../venv/bin/activate
fi

# Install dependencies
echo ""
echo "Checking Python dependencies..."
pip install -r requirements.txt --quiet

# Copy .env from RAG if it exists and doesn't exist in backend
if [ -f "$PROJECT_ROOT/RAG/.env" ] && [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Copying .env from RAG folder..."
    cp "$PROJECT_ROOT/RAG/.env" "$SCRIPT_DIR/.env"
fi

# Start the Flask server
echo ""
echo "========================================"
echo "Starting Flask API Server..."
echo "========================================"
echo ""
echo -e "${GREEN}API will be available at: http://127.0.0.1:5000${NC}"
echo ""
echo "Endpoints:"
echo "  - POST /api/upload    - Upload PDF and classify"
echo "  - POST /api/extract   - Extract title/abstract only"
echo "  - POST /api/classify  - Classify given title/abstract"
echo "  - GET  /health        - Health check"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py --host 127.0.0.1 --port 5000
