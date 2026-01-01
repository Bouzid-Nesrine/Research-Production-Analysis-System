# Complete System Deployment Guide
## RAG Classification System with Backend API

This guide covers hosting the complete Research Production Analysis System including:
- **RAG Classification Pipeline** (embeddings + LLM)
- **ChromaDB Vector Database** (taxonomy paths)
- **Flask Backend API** (REST endpoints)
- **GROBID Service** (PDF extraction)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React/Next.js)                      │
│                      Port: 3000/5173                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP Requests
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FLASK BACKEND API                             │
│                        Port: 5000                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ PDF Upload   │  │ Text Extract │  │ Classify     │          │
│  │ /upload      │  │ /extract     │  │ /classify    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  │
┌──────────────────┐  ┌──────────────────┐     │
│  GROBID          │  │  PDF Extractor   │     │
│  Docker:8070     │  │  (Python)        │     │
│  PDF→Text        │  │                  │     │
└──────────────────┘  └──────────────────┘     │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │   RAG PIPELINE               │
                                 │  ┌────────────────────────┐  │
                                 │  │ 1. Embedding Model     │  │
                                 │  │    all-MiniLM-L6-v2    │  │
                                 │  └───────────┬────────────┘  │
                                 │              ▼               │
                                 │  ┌────────────────────────┐  │
                                 │  │ 2. ChromaDB           │  │
                                 │  │    Vector Database     │  │
                                 │  │    4,523 paths         │  │
                                 │  └───────────┬────────────┘  │
                                 │              ▼               │
                                 │  ┌────────────────────────┐  │
                                 │  │ 3. LLM Classifier     │  │
                                 │  │    Gemini API          │  │
                                 │  │    (Multi-key)         │  │
                                 │  └────────────────────────┘  │
                                 └──────────────────────────────┘
```

---

## 📋 Prerequisites

### Required Software:
- ✅ **Docker** (for GROBID service)
- ✅ **Python 3.8+** (for backend and RAG)
- ✅ **Node.js 16+** (for frontend, if using)
- ✅ **Git** (for deployment)

### Required API Keys:
- ✅ **Google AI Studio API Keys** (4-5 keys recommended)
  - Get from: https://aistudio.google.com/app/apikey

### System Requirements:
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models and database
- **Network**: Internet access for API calls

---

## 🚀 Deployment Steps

### Step 1: Clone and Setup Project

```bash
# Navigate to project root
cd /home/zahra/Documents/4rth\ Year/NLP/Project/Research-Production-Analysis-System

# Check structure
ls -la
# Should see: backend/ RAG/ nlp-platform-ui/ Taxonomy\ Building/
```

### Step 2: Setup RAG Pipeline

```bash
cd RAG

# Install Python dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
nano .env  # Edit and add your API keys
```

**Edit `.env` file:**
```bash
# Add your Google API keys (comma-separated)
GOOGLE_API_KEYS=AIzaSy...key1,AIzaSy...key2,AIzaSy...key3,AIzaSy...key4

# Optional: Single key fallback
GOOGLE_API_KEY=AIzaSy...key1
```

**Initialize ChromaDB with taxonomy:**
```bash
# One-time setup: Build vector database
python setup_pipeline.py

# Verify setup
python -c "from rag_pipeline import RAGClassificationPipeline; p = RAGClassificationPipeline(auto_setup=True); print('✓ RAG Pipeline Ready')"
```

### Step 3: Setup Backend API

```bash
cd ../backend

# Install dependencies
pip install -r requirements.txt

# Start the backend (includes GROBID)
chmod +x start.sh
./start.sh
```

**What `start.sh` does:**
1. ✅ Checks Docker installation
2. ✅ Starts GROBID container (port 8070)
3. ✅ Installs Python dependencies
4. ✅ Initializes RAG pipeline
5. ✅ Starts Flask API (port 5000)

### Step 4: Verify Services

**Check GROBID:**
```bash
curl http://localhost:8070/api/isalive
# Should return: true
```

**Check Backend API:**
```bash
curl http://localhost:5000/health
# Should return: {"status": "healthy", ...}
```

**Check RAG Pipeline:**
```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Deep Learning for Image Classification",
    "abstract": "This paper presents a convolutional neural network..."
  }'
```

### Step 5: Setup Frontend (Optional)

```bash
cd ../nlp-platform-ui

# Install dependencies
npm install

# Start development server
npm run dev
# Opens on http://localhost:5173
```

---

## 🔧 Configuration Files

### 1. RAG Configuration (`RAG/config.py`)

```python
# Model Selection
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast embeddings
LLM_MODEL_NAME = "gemini-2.0-flash-exp"     # Stable, fast LLM

# RAG Parameters
RAG_CONFIG = {
    "top_k": 5,              # Number of paths to retrieve
    "temperature": 0.1,      # LLM creativity (lower = deterministic)
    "max_tokens": 150,       # Response length limit
    "max_retries": 3,        # Retry attempts on failure
}
```

### 2. Backend Configuration (`backend/app.py`)

```python
# Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000       # API port

# CORS origins (add your frontend URL)
CORS(app, origins=[
    "http://localhost:5173",     # Vite dev
    "http://localhost:3000",     # Next.js dev
    "https://yourdomain.com"     # Production
])

# File upload limits
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max
```

### 3. Environment Variables (`RAG/.env`)

```bash
# Google AI Studio API Keys (REQUIRED)
GOOGLE_API_KEYS=key1,key2,key3,key4

# Optional: ChromaDB path override
# CHROMA_DB_PATH=/custom/path/to/chroma_db

# Optional: Taxonomy path override
# TAXONOMY_PATH=/custom/path/to/taxonomy.json
```

---

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "services": {
    "grobid": true,
    "rag_pipeline": true
  }
}
```

### Upload PDF
```bash
POST /api/upload
Content-Type: multipart/form-data
Body: file=<PDF file>
```

### Extract Metadata
```bash
POST /api/extract
Content-Type: multipart/form-data
Body: file=<PDF file>
```
Response:
```json
{
  "title": "Paper Title",
  "abstract": "Paper abstract...",
  "extraction_time": 2.5
}
```

### Classify Article
```bash
POST /api/classify
Content-Type: application/json
```
Body:
```json
{
  "title": "Article title",
  "abstract": "Article abstract..."
}
```
Response:
```json
{
  "classification": {
    "path": "Natural Science > Computer Science > AI > ML > Deep Learning",
    "confidence": "High",
    "hierarchy": [...]
  },
  "metadata": {
    "retrieved_paths": [...],
    "retrieval_scores": [0.89, 0.85, ...],
    "inference_time": 6.2
  }
}
```

### Complete Pipeline (Upload + Extract + Classify)
```bash
POST /api/process
Content-Type: multipart/form-data
Body: file=<PDF file>
```

---

## 🗄️ Database Management

### ChromaDB (Vector Database)

**Location:**
```bash
RAG/chroma_db/
```

**Reset Database:**
```bash
cd RAG
python setup_pipeline.py --reset
```

**Check Database Status:**
```python
from vector_db_manager import VectorDBManager

db = VectorDBManager(db_path="chroma_db")
db.initialize_collection()
print(f"Total paths: {db.collection.count()}")
```

**Backup Database:**
```bash
# Backup entire database
tar -czf chroma_db_backup_$(date +%Y%m%d).tar.gz RAG/chroma_db/

# Restore from backup
tar -xzf chroma_db_backup_YYYYMMDD.tar.gz -C RAG/
```

---

## 🔐 Security Considerations

### 1. API Keys
```bash
# NEVER commit .env to git
echo ".env" >> .gitignore

# Use environment variables in production
export GOOGLE_API_KEYS="key1,key2,key3"
```

### 2. CORS Configuration
```python
# backend/app.py
# Only allow specific origins in production
CORS(app, origins=["https://yourdomain.com"])
```

### 3. File Upload Security
- Maximum file size: 10MB (configurable)
- Allowed extensions: PDF only
- Files are sanitized with `secure_filename()`
- Temporary storage in `backend/uploads/`

### 4. Rate Limiting (Recommended)
```bash
pip install flask-limiter
```
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=get_remote_address)

@limiter.limit("10 per minute")
@app.route('/api/classify', methods=['POST'])
def classify():
    ...
```

---

## 🚦 Production Deployment

### Option 1: Single Server (Simple)

```bash
# 1. Install dependencies
cd backend && pip install -r requirements.txt
cd ../RAG && pip install -r requirements.txt

# 2. Start services in background
cd ../backend
nohup ./start.sh > backend.log 2>&1 &

# 3. Check logs
tail -f backend.log
```

### Option 2: Docker Compose (Recommended)

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  grobid:
    image: lfoppiano/grobid:0.8.1
    ports:
      - "8070:8070"
    volumes:
      - grobid-data:/opt/grobid/data

  backend:
    build: ./backend
    ports:
      - "5000:5000"
    depends_on:
      - grobid
    environment:
      - GOOGLE_API_KEYS=${GOOGLE_API_KEYS}
    volumes:
      - ./RAG:/app/RAG
      - ./backend/uploads:/app/uploads
      - chroma-db:/app/RAG/chroma_db

volumes:
  grobid-data:
  chroma-db:
```

Deploy:
```bash
docker-compose up -d
docker-compose logs -f
```

### Option 3: Cloud Deployment (AWS/GCP/Azure)

**Requirements:**
- VM with 4GB+ RAM
- Docker installed
- Ports 5000, 8070 open
- Domain name (optional)

**Steps:**
```bash
# 1. SSH to server
ssh user@your-server.com

# 2. Clone repository
git clone <your-repo-url>
cd Research-Production-Analysis-System

# 3. Setup environment
cp RAG/.env.example RAG/.env
nano RAG/.env  # Add API keys

# 4. Start services
cd backend && ./start.sh

# 5. Setup nginx (optional)
sudo apt install nginx
# Configure reverse proxy to port 5000
```

---

## 📊 Monitoring and Logging

### Log Files

**Backend Logs:**
```bash
backend/logs/app.log
```

**RAG Pipeline Logs:**
```bash
RAG/logs/rag_classification.log
```

**GROBID Logs:**
```bash
docker logs grobid
```

### Health Monitoring

**Monitor API:**
```bash
# Check every minute
watch -n 60 'curl -s http://localhost:5000/health | jq'
```

**Monitor API Key Status:**
```bash
curl http://localhost:5000/api/key-status
```

**Monitor Performance:**
```python
# In RAG/logs/
tail -f rag_classification.log | grep "Classification time"
```

---

## 🐛 Troubleshooting

### Issue: "GROBID not responding"
```bash
# Check GROBID container
docker ps | grep grobid

# Restart GROBID
docker restart grobid

# Check logs
docker logs grobid --tail 50
```

### Issue: "RAG pipeline initialization failed"
```bash
# Check ChromaDB
ls -la RAG/chroma_db/

# Rebuild database
cd RAG && python setup_pipeline.py --reset

# Verify setup
python -c "from rag_pipeline import RAGClassificationPipeline; RAGClassificationPipeline(auto_setup=True)"
```

### Issue: "API quota exceeded"
```bash
# Check API key rotation
python -c "from llm_classifier import LLMClassifier; print(LLMClassifier().get_api_key_status())"

# Add more keys in .env
GOOGLE_API_KEYS=key1,key2,key3,key4,key5
```

### Issue: "Slow classification"
- **Normal**: 5-10 seconds per article
- **Check**: Network latency to Google API
- **Solution**: Use more API keys for parallelism

### Issue: "Port already in use"
```bash
# Check what's using port 5000
lsof -i :5000

# Change port in backend/app.py
PORT = 5001  # Use different port
```

---

## 🧪 Testing

### Test RAG Pipeline
```bash
cd RAG
python evaluate_accuracy.py
```

### Test Backend API
```bash
cd backend

# Test health
curl http://localhost:5000/health

# Test classification
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"title":"Test","abstract":"Test abstract"}'
```

### Load Testing
```bash
pip install locust

# Create locustfile.py
# Run load test
locust -f locustfile.py --host=http://localhost:5000
```

---

## 📦 Deployment Checklist

- [ ] **Python dependencies installed** (backend + RAG)
- [ ] **Docker installed and running**
- [ ] **GROBID container started** (port 8070)
- [ ] **ChromaDB initialized** (4,523 paths)
- [ ] **API keys configured** in `RAG/.env`
- [ ] **Backend API running** (port 5000)
- [ ] **Health check passes** `/health`
- [ ] **Test classification works** `/api/classify`
- [ ] **Logs configured** and accessible
- [ ] **CORS configured** for frontend
- [ ] **Security measures** in place
- [ ] **Backup strategy** for ChromaDB
- [ ] **Monitoring setup** for uptime
- [ ] **Documentation** for team

---

## 📚 Additional Resources

- **API Documentation**: `backend/README.md`
- **RAG Documentation**: `RAG/README.md`
- **API Key Rotation**: `RAG/API_KEY_ROTATION.md`
- **Accuracy Testing**: `RAG/ACCURACY_TESTING_GUIDE.md`

---

## 🆘 Getting Help

1. **Check logs**: `backend/logs/` and `RAG/logs/`
2. **Test components**: Use test scripts in `RAG/test_*.py`
3. **Verify services**: `curl http://localhost:5000/health`
4. **Review documentation**: Check README files

---

## 🎉 Quick Start Command

```bash
# From project root, start everything:
cd backend && ./start.sh
```

This single command:
- ✅ Starts GROBID
- ✅ Initializes ChromaDB
- ✅ Loads RAG pipeline
- ✅ Starts Flask API

**Access at**: `http://localhost:5000`

**Frontend connects to**: `http://localhost:5000/api/*`
