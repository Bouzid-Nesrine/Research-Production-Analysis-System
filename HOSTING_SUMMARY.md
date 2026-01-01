# 🎯 Complete Hosting Guide - Summary

## ✅ What You Have Now

I've created a **complete deployment system** for your RAG classification project with:

### 📁 New Files Created:

1. **`start_system.sh`** - One-command startup (ALL services)
2. **`stop_system.sh`** - Clean shutdown script
3. **`DEPLOYMENT_GUIDE.md`** - Complete technical deployment guide
4. **`QUICKSTART_DEPLOYMENT.md`** - Quick start for deployment

---

## 🚀 How to Host the Complete System

### **SUPER SIMPLE - 2 Commands:**

```bash
# 1. Make sure you have API keys in RAG/.env
nano RAG/.env

# 2. Start everything
./start_system.sh
```

**That's it!** 🎉

The script will:
- ✅ Check Docker & Python
- ✅ Start GROBID container (port 8070)
- ✅ Initialize ChromaDB (4,523 taxonomy paths)
- ✅ Load RAG pipeline (embeddings + LLM)
- ✅ Start Flask Backend API (port 5000)

---

## 🏗️ System Components

### What Gets Hosted:

```
┌─────────────────────────────────────────────────────────┐
│                   YOUR FRONTEND                          │
│              (React/Next.js on port 3000/5173)          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              FLASK BACKEND API - Port 5000               │
│  • /api/upload    - Upload PDF                          │
│  • /api/extract   - Extract title/abstract              │
│  • /api/classify  - Classify article                    │
│  • /api/process   - Complete pipeline                   │
└────────┬────────────────────────┬────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────────────────────┐
│  GROBID         │    │      RAG PIPELINE                │
│  Port: 8070     │    │  ┌────────────────────────┐      │
│  (Docker)       │    │  │ 1. Embedding Model     │      │
│                 │    │  │    all-MiniLM-L6-v2    │      │
│  PDF→Text       │    │  └──────────┬─────────────┘      │
└─────────────────┘    │             ▼                    │
                       │  ┌────────────────────────┐      │
                       │  │ 2. ChromaDB            │      │
                       │  │    4,523 paths         │      │
                       │  └──────────┬─────────────┘      │
                       │             ▼                    │
                       │  ┌────────────────────────┐      │
                       │  │ 3. LLM Classifier      │      │
                       │  │    Google Gemini       │      │
                       │  │    Multi-key rotation  │      │
                       │  └────────────────────────┘      │
                       └──────────────────────────────────┘
```

---

## 📍 What's Hosted Where

| Component | Location | Port | Purpose |
|-----------|----------|------|---------|
| **GROBID** | Docker Container | 8070 | PDF text extraction |
| **Backend API** | Flask Python Server | 5000 | REST API endpoints |
| **ChromaDB** | `RAG/chroma_db/` | N/A | Vector database (local files) |
| **RAG Pipeline** | Python Process | N/A | Classification logic |
| **Embeddings** | In-memory | N/A | Sentence transformers model |
| **LLM** | Google Cloud | API | Gemini classification |

---

## 🔌 API Endpoints (Port 5000)

### Health Check
```bash
GET http://localhost:5000/health
```

### Upload PDF
```bash
POST http://localhost:5000/api/upload
Content-Type: multipart/form-data
```

### Classify Article
```bash
POST http://localhost:5000/api/classify
Content-Type: application/json

{
  "title": "Article Title",
  "abstract": "Article abstract..."
}
```

### Complete Pipeline (Upload + Extract + Classify)
```bash
POST http://localhost:5000/api/process
Content-Type: multipart/form-data
```

---

## 📊 Data Storage

### ChromaDB (Vector Database)
- **Location**: `RAG/chroma_db/`
- **Content**: 4,523 embedded taxonomy paths
- **Size**: ~15MB
- **Format**: Persistent local storage
- **Backup**: `tar -czf chroma_backup.tar.gz RAG/chroma_db/`

### Uploaded PDFs
- **Location**: `backend/uploads/`
- **Cleanup**: Temporary (can be cleared)

### Logs
- **Backend**: `backend/logs/app.log`
- **RAG**: `RAG/logs/rag_classification.log`

---

## ⚙️ Configuration

### Required Environment Variables (`RAG/.env`)
```bash
# REQUIRED: Your Google API keys (comma-separated)
GOOGLE_API_KEYS=key1,key2,key3,key4

# Optional: Single key fallback
GOOGLE_API_KEY=key1
```

### Model Configuration (`RAG/config.py`)
```python
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast CPU embeddings
LLM_MODEL_NAME = "gemini-2.0-flash-exp"     # Stable Google model

RAG_CONFIG = {
    "top_k": 5,              # Retrieved paths
    "temperature": 0.1,      # LLM randomness
    "max_tokens": 150,       # Response length
}
```

---

## 🌐 Deployment Options

### Option 1: Local Development (Current)
```bash
./start_system.sh
# Access at: http://localhost:5000
```

### Option 2: Cloud Server (AWS/GCP/Azure)
```bash
# 1. SSH to server
ssh user@your-server.com

# 2. Clone repo
git clone <your-repo-url>

# 3. Setup
cd Research-Production-Analysis-System
nano RAG/.env  # Add API keys

# 4. Start
./start_system.sh

# Access at: http://your-server.com:5000
```

### Option 3: Docker Compose (Recommended for Production)
- See `DEPLOYMENT_GUIDE.md` for docker-compose.yml
- Containerizes everything for easy deployment
- Persistent volumes for ChromaDB

---

## 🧪 Testing

### Test Backend API
```bash
curl http://localhost:5000/health
```

### Test Classification
```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Deep Learning for Computer Vision",
    "abstract": "This paper presents a CNN for image classification..."
  }'
```

### Test RAG Accuracy
```bash
cd RAG
python3 evaluate_accuracy.py
```

---

## 📈 Performance

### Expected Metrics:
- **Retrieval Time**: 10-50ms (ChromaDB search)
- **Classification Time**: 5-10 seconds (with LLM)
- **Total Time per Article**: ~6-10 seconds
- **Throughput**: ~360-600 articles/hour
- **With 4 API keys**: ~1,440-2,400 articles/hour

### Resource Usage:
- **RAM**: ~2-3GB (embeddings + ChromaDB)
- **CPU**: Moderate (embeddings on CPU)
- **Disk**: ~15MB (ChromaDB) + uploads
- **Network**: API calls to Google Gemini

---

## 🔒 Security

### Implemented:
- ✅ File type validation (PDF only)
- ✅ File size limits (10MB max)
- ✅ Secure filename handling
- ✅ CORS configured for specific origins
- ✅ API key rotation for redundancy
- ✅ Environment variable for secrets

### Recommended for Production:
- Add rate limiting
- Enable HTTPS
- Add authentication
- Regular backups
- Monitor logs

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `QUICKSTART_DEPLOYMENT.md` | Quick start guide |
| `DEPLOYMENT_GUIDE.md` | Complete technical guide |
| `RAG/README.md` | RAG pipeline documentation |
| `backend/README.md` | Backend API documentation |
| `RAG/API_KEY_ROTATION.md` | Multi-key setup guide |

---

## 🆘 Quick Troubleshooting

### "Docker not found"
```bash
sudo apt install docker.io  # Ubuntu
brew install docker         # macOS
```

### "Permission denied"
```bash
chmod +x start_system.sh
```

### "Port 5000 in use"
```bash
# Kill existing process
lsof -i :5000
kill -9 <PID>
```

### "GROBID not starting"
```bash
# Restart Docker
docker restart grobid
```

### "ChromaDB error"
```bash
# Rebuild database
cd RAG && python3 setup_pipeline.py --reset
```

---

## ✅ Deployment Checklist

- [ ] Docker installed and running
- [ ] Python 3.8+ installed
- [ ] API keys added to `RAG/.env`
- [ ] Run `./start_system.sh`
- [ ] Check `http://localhost:5000/health` → should return `{"status": "healthy"}`
- [ ] Test classification endpoint
- [ ] Verify GROBID: `http://localhost:8070/api/isalive` → should return `true`
- [ ] Verify ChromaDB: Should have 4,523 paths
- [ ] Check logs are being created
- [ ] Test with frontend connection

---

## 🎯 Next Steps

1. **Start the system**: `./start_system.sh`
2. **Test endpoints**: Use curl or Postman
3. **Connect frontend**: Point to `http://localhost:5000/api/*`
4. **Monitor logs**: `tail -f backend/logs/app.log`
5. **Deploy to cloud**: Follow `DEPLOYMENT_GUIDE.md`

---

## 📞 Quick Commands

```bash
# Start everything
./start_system.sh

# Stop everything
./stop_system.sh

# Check health
curl http://localhost:5000/health

# Test classification
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"title":"Test","abstract":"Test abstract about AI"}'

# View logs
tail -f backend/logs/app.log
tail -f RAG/logs/rag_classification.log
docker logs grobid

# Check API keys
cd RAG && python3 -c "from llm_classifier import LLMClassifier; print(LLMClassifier().get_api_key_status())"
```

---

## 🎉 You're Ready!

Your complete RAG classification system is ready to host!

**Just run:**
```bash
./start_system.sh
```

Then access your API at: **http://localhost:5000** 🚀
