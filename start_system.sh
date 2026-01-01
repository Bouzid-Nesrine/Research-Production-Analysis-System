#!/bin/bash

# Complete System Startup Script
# Starts GROBID, Backend API, and RAG Pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║  Research Production Analysis System - Startup        ║"
echo "║  RAG + Backend API + GROBID                           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to check command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=0

    echo -e "${YELLOW}Waiting for $name to be ready...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $name is ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
        echo -n "."
    done
    
    echo -e "${RED}✗ $name failed to start${NC}"
    return 1
}

# Step 1: Check Prerequisites
echo -e "\n${BLUE}[1/5] Checking Prerequisites${NC}"

if command_exists docker; then
    echo -e "${GREEN}✓ Docker installed${NC}"
else
    echo -e "${RED}✗ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

if docker info &> /dev/null; then
    echo -e "${GREEN}✓ Docker daemon running${NC}"
else
    echo -e "${RED}✗ Docker daemon not running. Please start Docker.${NC}"
    exit 1
fi

if command_exists python3; then
    echo -e "${GREEN}✓ Python3 installed${NC}"
else
    echo -e "${RED}✗ Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Step 2: Start GROBID
echo -e "\n${BLUE}[2/5] Starting GROBID Service${NC}"

if docker ps --format '{{.Names}}' | grep -q "^grobid$"; then
    echo -e "${GREEN}✓ GROBID already running${NC}"
else
    echo -e "${YELLOW}Starting GROBID container...${NC}"
    
    # Check if container exists but stopped
    if docker ps -a --format '{{.Names}}' | grep -q "^grobid$"; then
        docker start grobid
    else
        docker run -d \
            --name grobid \
            -p 8070:8070 \
            --restart unless-stopped \
            lfoppiano/grobid:0.8.1
    fi
    
    # Wait for GROBID to be ready
    wait_for_service "http://localhost:8070/api/isalive" "GROBID"
fi

# Step 3: Setup RAG Pipeline
echo -e "\n${BLUE}[3/5] Setting up RAG Pipeline${NC}"

cd "$SCRIPT_DIR/RAG"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file not found. Creating from template...${NC}"
    cp .env.example .env
    echo -e "${RED}✗ Please edit RAG/.env and add your Google API keys${NC}"
    echo -e "${RED}  Then run this script again.${NC}"
    exit 1
fi

# Check if ChromaDB exists
if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
    echo -e "${YELLOW}ChromaDB not initialized. Running setup...${NC}"
    python3 setup_pipeline.py
else
    echo -e "${GREEN}✓ ChromaDB already initialized${NC}"
fi

# Install RAG dependencies
echo -e "${YELLOW}Installing RAG dependencies...${NC}"
pip3 install -q -r requirements.txt
echo -e "${GREEN}✓ RAG dependencies installed${NC}"

# Test RAG pipeline
echo -e "${YELLOW}Testing RAG pipeline...${NC}"
python3 -c "from rag_pipeline import RAGClassificationPipeline; RAGClassificationPipeline(auto_setup=True); print('✓ RAG Pipeline Ready')" || {
    echo -e "${RED}✗ RAG pipeline test failed${NC}"
    exit 1
}

# Step 4: Setup Backend
echo -e "\n${BLUE}[4/5] Setting up Backend API${NC}"

cd "$SCRIPT_DIR/backend"

# Install backend dependencies
echo -e "${YELLOW}Installing backend dependencies...${NC}"
pip3 install -q -r requirements.txt
echo -e "${GREEN}✓ Backend dependencies installed${NC}"

# Create uploads directory
mkdir -p uploads
echo -e "${GREEN}✓ Uploads directory ready${NC}"

# Step 5: Start Backend Server
echo -e "\n${BLUE}[5/5] Starting Backend Server${NC}"

export FLASK_APP=app.py
export FLASK_ENV=production

echo -e "${YELLOW}Starting Flask server on port 5000...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}\n"

# Start Flask
python3 -m flask run --host=0.0.0.0 --port=5000 &
FLASK_PID=$!

# Wait for Flask to be ready
sleep 3
wait_for_service "http://localhost:5000/health" "Backend API"

# Summary
echo -e "\n${GREEN}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║                 🎉 System Ready!                      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}Services running:${NC}"
echo -e "  • GROBID:      http://localhost:8070"
echo -e "  • Backend API: http://localhost:5000"
echo -e "  • ChromaDB:    RAG/chroma_db/"

echo -e "\n${GREEN}API Endpoints:${NC}"
echo -e "  • Health:      GET  http://localhost:5000/health"
echo -e "  • Upload PDF:  POST http://localhost:5000/api/upload"
echo -e "  • Extract:     POST http://localhost:5000/api/extract"
echo -e "  • Classify:    POST http://localhost:5000/api/classify"
echo -e "  • Process:     POST http://localhost:5000/api/process"

echo -e "\n${YELLOW}Logs:${NC}"
echo -e "  • Backend: backend/logs/app.log"
echo -e "  • RAG:     RAG/logs/rag_classification.log"
echo -e "  • GROBID:  docker logs grobid"

echo -e "\n${YELLOW}To stop services:${NC}"
echo -e "  • Press Ctrl+C to stop Flask"
echo -e "  • Run: docker stop grobid"

echo -e "\n${BLUE}Monitoring server... (Ctrl+C to stop)${NC}\n"

# Keep script running and forward Flask logs
wait $FLASK_PID
