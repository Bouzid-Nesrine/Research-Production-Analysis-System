# Quick Start Guide - Complete System Deployment

## 🚀 Deploy Everything in 2 Commands

### 1. Start the System
```bash
chmod +x start_system.sh
./start_system.sh
```

This automatically:
- ✅ Checks Docker and Python
- ✅ Starts GROBID (port 8070)
- ✅ Initializes ChromaDB with taxonomy
- ✅ Loads RAG pipeline
- ✅ Starts Flask API (port 5000)

### 2. Stop the System
```bash
chmod +x stop_system.sh
./stop_system.sh
```

---

## ⚡ What You Get

### Services Running:
- **GROBID**: http://localhost:8070 (PDF extraction)
- **Backend API**: http://localhost:5000 (REST endpoints)
- **ChromaDB**: RAG/chroma_db/ (vector database)
- **RAG Pipeline**: Embeddings + LLM classification

### API Endpoints:
```bash
# Health check
curl http://localhost:5000/health

# Classify article
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Deep Learning for Image Classification",
    "abstract": "This paper presents a CNN architecture..."
  }'

# Upload and process PDF
curl -X POST http://localhost:5000/api/process \
  -F "file=@paper.pdf"
```

---

## 📋 Before First Run

### 1. Add API Keys (Required)
```bash
# Edit this file
nano RAG/.env

# Add your Google API keys (comma-separated)
GOOGLE_API_KEYS=AIzaSy...key1,AIzaSy...key2,AIzaSy...key3
```

### 2. Install Docker (if not installed)
```bash
# Ubuntu/Debian
sudo apt install docker.io

# macOS
brew install --cask docker

# Start Docker
sudo systemctl start docker  # Linux
# Or open Docker Desktop on macOS/Windows
```

### 3. Install Python Dependencies (handled automatically)
The startup script installs everything needed.

---

## 📁 System Components

### 1. **RAG Pipeline** (`RAG/`)
- **Embedding Model**: all-MiniLM-L6-v2 (CPU-friendly)
- **Vector DB**: ChromaDB with 4,523 taxonomy paths
- **LLM**: Google Gemini (multi-key rotation)
- **Retrieval**: 10-50ms semantic search
- **Classification**: 5-10s per article

### 2. **Backend API** (`backend/`)
- **Framework**: Flask (Python)
- **Port**: 5000
- **Endpoints**: Upload, Extract, Classify, Process
- **Features**: CORS, file validation, caching

### 3. **GROBID Service** (Docker)
- **Port**: 8070
- **Purpose**: PDF → Text extraction
- **Image**: lfoppiano/grobid:0.8.1

---

## 🔧 Configuration

### RAG Configuration (`RAG/config.py`)
```python
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-2.0-flash-exp"

RAG_CONFIG = {
    "top_k": 5,           # Paths to retrieve
    "temperature": 0.1,   # LLM creativity
    "max_tokens": 150,    # Response length
    "max_retries": 3,     # API retry attempts
}
```

### Backend Configuration (`backend/app.py`)
```python
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000       # API port
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max upload
```

---

## 📊 Monitoring

### Check Service Health
```bash
# Backend API
curl http://localhost:5000/health

# GROBID
curl http://localhost:8070/api/isalive

# ChromaDB path count
python3 -c "from RAG.vector_db_manager import VectorDBManager; db = VectorDBManager(db_path='RAG/chroma_db'); db.initialize_collection(); print(f'Paths: {db.collection.count()}')"
```

### View Logs
```bash
# Backend logs
tail -f backend/logs/app.log

# RAG logs
tail -f RAG/logs/rag_classification.log

# GROBID logs
docker logs grobid --tail 50
```

### API Key Status
```bash
python3 -c "from RAG.llm_classifier import LLMClassifier; import json; print(json.dumps(LLMClassifier().get_api_key_status(), indent=2))"
```

---

## 🐛 Troubleshooting

### "Docker not found"
Install Docker: https://docs.docker.com/get-docker/

### "Permission denied" on scripts
```bash
chmod +x start_system.sh stop_system.sh
```

### "GROBID not starting"
```bash
# Check Docker is running
docker info

# Restart Docker
sudo systemctl restart docker  # Linux
```

### "API key error"
```bash
# Verify .env file exists
cat RAG/.env

# Should show: GOOGLE_API_KEYS=...
```

### "ChromaDB initialization failed"
```bash
# Rebuild database
cd RAG
python3 setup_pipeline.py --reset
```

### "Port 5000 already in use"
```bash
# Find what's using it
lsof -i :5000

# Kill the process or change port in backend/app.py
```

---

## 🎯 Testing

### Test RAG Accuracy
```bash
cd RAG
python3 evaluate_accuracy.py
```

### Test API Endpoints
```bash
# Test classification
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"title":"Test Article","abstract":"This is a test abstract about machine learning and neural networks."}'
```

### Test with Sample PDF
```bash
curl -X POST http://localhost:5000/api/process \
  -F "file=@sample_paper.pdf"
```

---

## 📦 Deployment Checklist

- [ ] Docker installed and running
- [ ] Python 3.8+ installed
- [ ] API keys added to `RAG/.env`
- [ ] Run `./start_system.sh`
- [ ] Health check passes
- [ ] Test classification works
- [ ] ChromaDB initialized (4,523 paths)
- [ ] GROBID responding
- [ ] Logs accessible

---

## 🌐 Production Deployment

### Cloud Server Setup

1. **Provision Server** (AWS/GCP/Azure)
   - 4GB+ RAM
   - Ubuntu 20.04+
   - Ports 5000, 8070 open

2. **Clone Repository**
   ```bash
   git clone <your-repo>
   cd Research-Production-Analysis-System
   ```

3. **Setup Environment**
   ```bash
   # Add API keys
   nano RAG/.env
   
   # Start system
   ./start_system.sh
   ```

4. **Setup Reverse Proxy** (Optional)
   ```bash
   sudo apt install nginx
   
   # Configure nginx to proxy port 5000
   # Then access at yourdomain.com
   ```

5. **Enable Auto-start** (systemd)
   ```bash
   sudo nano /etc/systemd/system/research-api.service
   
   # Add service configuration
   sudo systemctl enable research-api
   sudo systemctl start research-api
   ```

---

## 📚 Additional Documentation

- **Complete Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **RAG Documentation**: `RAG/README.md`
- **Backend API**: `backend/README.md`
- **API Key Rotation**: `RAG/API_KEY_ROTATION.md`
- **Accuracy Testing**: `RAG/ACCURACY_TESTING_GUIDE.md`

---

## 🆘 Need Help?

1. **Check logs**: `backend/logs/` and `RAG/logs/`
2. **Verify services**: `curl http://localhost:5000/health`
3. **Test components**: `cd RAG && python3 test_5_rag_pipeline.py`
4. **Review documentation**: See guides above

---

## 🎉 You're All Set!

Your complete research classification system is now running:

```
Frontend → Backend API (5000) → RAG Pipeline → Gemini API
                ↓
             GROBID (8070)
                ↓
           ChromaDB (4,523 paths)
```

**Access the API**: http://localhost:5000

**Next Steps**:
1. Connect your frontend to http://localhost:5000/api/*
2. Test with sample PDFs
3. Monitor performance in logs
4. Add more API keys for higher throughput
