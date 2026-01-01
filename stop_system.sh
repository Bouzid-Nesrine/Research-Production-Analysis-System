#!/bin/bash

# System Shutdown Script
# Stops all running services

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping Research Production Analysis System...${NC}\n"

# Stop Flask
echo -e "${YELLOW}Stopping Flask backend...${NC}"
pkill -f "flask run" || echo -e "${YELLOW}Flask not running${NC}"
echo -e "${GREEN}✓ Flask stopped${NC}"

# Stop GROBID
echo -e "\n${YELLOW}Stopping GROBID container...${NC}"
if docker ps --format '{{.Names}}' | grep -q "^grobid$"; then
    docker stop grobid
    echo -e "${GREEN}✓ GROBID stopped${NC}"
else
    echo -e "${YELLOW}GROBID not running${NC}"
fi

echo -e "\n${GREEN}All services stopped${NC}"
echo -e "${YELLOW}Note: ChromaDB data is preserved in RAG/chroma_db/${NC}"
