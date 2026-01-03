# Research Production Analysis System

A comprehensive platform for scientific article classification using RAG (Retrieval-Augmented Generation) with fine-tuned SciBERT models and hierarchical taxonomy.

---

## 🎯 Overview

This system provides an end-to-end solution for classifying research articles into a hierarchical taxonomy of 1,449 scientific domains. It combines:

- **Fine-tuned SciBERT Model**: Domain-specific classification with LoRA adaptation (862 classes, F1=27%)
- **RAG Enhancement**: Retrieval + re-ranking pipeline improving accuracy to **34.54%** (+27.9% relative improvement)
- **Web Platform**: Interactive UI for article upload, PDF parsing, and classification
- **REST API**: Backend service for integration with other systems

---

## � Live Demo

**Try the platform now**: [https://research-analytics.netlify.app/](https://research-analytics.netlify.app/)

The system is hosted and ready to use. Upload your research papers and get instant classification results.

---

## �🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Application (React)                   │
│                         Port: 5173                           │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP Requests
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Backend API (Flask)                         │
│                         Port: 5000                           │
├─────────────────────────────────────────────────────────────┤
│  • PDF Parsing (GROBID)                                     │
│  • Article Processing                                        │
│  • RAG Pipeline Orchestration                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              RAG Classification Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Semantic Retrieval (ChromaDB)                     │
│    • Embedding: all-MiniLM-L6-v2                           │
│    • Top-K candidates (K=5)                                 │
│    • Time: ~0.024s                                          │
│                                                              │
│  Stage 2: Neural Re-ranking (SciBERT + LoRA)               │
│    • Fine-tuned model scoring                               │
│    • Score fusion (60% model + 40% retrieval)              │
│    • Time: ~0.83s                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 16+** and npm
- **8GB+ RAM**
- **Virtual environment** (recommended)

### 1. Clone Repository

```bash
git clone <repository-url>
cd Research-Production-Analysis-System
```

### 2. Setup Python Environment

```bash
# Create virtual environment
python -m venv ai_env

# Activate virtual environment
# Linux/Mac:
source ai_env/bin/activate
# Windows:
ai_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup RAG Database (One-time)

```bash
cd RAG
python setup_pipeline.py
cd ..
```

This will:
- Parse taxonomy into 1,449 paths
- Generate embeddings
- Initialize ChromaDB vector database

### 4. Run Backend Server

```bash
cd backend
python app.py
```

Backend will start on: **http://127.0.0.1:5000**

**Expected output:**
```
✓ RAG pipeline initialized successfully
✓ GROBID is running
✓ Starting API server on 127.0.0.1:5000
```

### 5. Run Frontend (New Terminal)

```bash
# Open new terminal
cd nlp-platform-ui
npm install  # First time only
npm run dev
```

Frontend will start on: **http://localhost:5173**

### 6. Access the Platform

**Option 1: Use Hosted Version**  
Visit: **https://research-analytics.netlify.app/**

**Option 2: Run Locally**  
Open your browser and navigate to: **http://localhost:5173**

---

## 📁 Project Structure

```
Research-Production-Analysis-System/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── backend/                           # Flask REST API
│   ├── app.py                        # Main server application
│   ├── pdf_parser.py                 # GROBID PDF processing
│   └── routes/                       # API endpoints
│
├── nlp-platform-ui/                  # React frontend
│   ├── src/
│   │   ├── components/              # UI components
│   │   ├── services/                # API client
│   │   └── App.tsx                  # Main application
│   ├── package.json
│   └── vite.config.ts
│
├── RAG/                              # RAG classification system
│   ├── README.md                    # RAG-specific documentation
│   ├── EVALUATION_REPORT.md         # Performance analysis
│   ├── rag_pipeline.py              # Main pipeline
│   ├── local_model_classifier.py    # Fine-tuned SciBERT classifier
│   ├── vector_db_manager.py         # ChromaDB interface
│   ├── taxonomy_parser.py           # Taxonomy processing
│   ├── setup_pipeline.py            # Database initialization
│   ├── evaluate_rag_accuracy.py     # Evaluation script
│   ├── best_models/
│   │   └── scibert_lora_final/      # Fine-tuned model (862 classes)
│   ├── chroma_db/                   # Vector database (created after setup)
│   └── evaluation_results/          # Test results
│
├── fine_tuning/                      # Model training
│   ├── train_model.py               # Training script
│   ├── prepare_data.py              # Data preparation
│   ├── inference.py                 # Model inference
│   └── final_model/                 # Alternative model (1393 classes)
│
├── Taxonomy Building/                # Taxonomy construction
│   ├── final_combined_taxonomy.json # Complete taxonomy
│   ├── preprocess_taxonomy.py       # Taxonomy preprocessing
│   └── taxonomy_propmt.py           # LLM prompt for merging
│
├── Data Collection/                  # Article data collection
├── Data Preprocessing/               # Data cleaning
├── Annotation/                       # Manual annotation tools
└── evaluation/                       # Evaluation metrics
```

---

## 🎯 Key Features

### 1. **Intelligent Classification**
- **RAG-Enhanced**: 34.54% accuracy (27.9% improvement over baseline)
- **Two-Stage Pipeline**: Semantic retrieval + neural re-ranking
- **Fine-tuned SciBERT**: Domain-specific knowledge from 862 scientific classes

### 2. **PDF Processing**
- Upload research papers (PDF format)
- Automatic text extraction using GROBID
- Title and abstract parsing

### 3. **Interactive Web Interface**
- Drag-and-drop PDF upload
- Real-time classification results
- Hierarchical taxonomy visualization
- Confidence scores and reasoning

### 4. **REST API**
```bash
# Classify article
POST /api/classify
{
  "title": "Article title",
  "abstract": "Article abstract"
}

# Response
{
  "classification": {
    "path": "Natural Science > Computer Science > AI > Machine Learning",
    "confidence": "High",
    "confidence_score": 0.82,
    "model_score": 0.85,
    "reasoning": "..."
  }

```

---

## 🔧 Configuration

### Backend Configuration

Edit `backend/config.py`:
```python
# Server settings
HOST = "127.0.0.1"
PORT = 5000

# GROBID settings
GROBID_URL = "http://localhost:8070"
```

### RAG Configuration

Edit `RAG/config.py`:
```python
RAG_CONFIG = {
    "top_k": 5,              # Number of candidates to retrieve
    "model_weight": 0.6,     # Weight for model score
    "retrieval_weight": 0.4  # Weight for retrieval similarity
}
```

---

## 📊 Performance Metrics

### Classification Performance
| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 34.54% (76/300 correct) |
| **Baseline (No RAG)** | 27.00% (F1 score) |
| **Improvement** | +7.54% absolute (+27.9% relative) |
| **Retrieval Recall@5** | 41.00% |
| **Re-ranking Accuracy** | 61.8% (when correct in top-5) |

### Speed Performance
| Stage | Time |
|-------|------|
| Retrieval | 0.024s |
| Re-ranking | 0.830s |
| **Total** | **0.854s** |

**See [RAG/EVALUATION_REPORT.md](RAG/EVALUATION_REPORT.md) for detailed analysis**

---

## 🧪 Testing & Evaluation

### Run Evaluation on Test Set

```bash
cd RAG
python evaluate_rag_accuracy.py
```

This will:
- Load 300 test articles
- Run RAG classification
- Generate detailed metrics
- Save results to `evaluation_results/`

### Generate Performance Figures

```bash
cd RAG
python demo_for_report.py
```

Generates:
- Performance comparison chart
- System architecture diagram
- Saved to `figures/`

---

## 🔍 API Endpoints

### Backend API (Port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/classify` | POST | Classify article by title/abstract |
| `/api/upload-pdf` | POST | Upload and classify PDF |
| `/api/health` | GET | Health check |
| `/api/taxonomy` | GET | Get full taxonomy structure |

### Example: Classify Article

```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Deep Learning for Medical Image Segmentation",
    "abstract": "This paper presents a novel CNN architecture..."
  }'
```

---

## 🛠️ Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# RAG tests
cd RAG
python test_rag_pipeline.py
```

### Adding New Features

1. **Backend**: Add routes in `backend/routes/`
2. **Frontend**: Add components in `nlp-platform-ui/src/components/`
3. **RAG**: Modify pipeline in `RAG/rag_pipeline.py`

---

## 🐛 Troubleshooting

### Backend not starting

```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill process if needed
kill -9 <PID>

# Restart backend
cd backend
python app.py
```

### Frontend build errors

```bash
# Clear node modules and reinstall
cd nlp-platform-ui
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### ChromaDB errors

```bash
# Rebuild vector database
cd RAG
python setup_pipeline.py --reset
```

### GROBID not available

The system will work without GROBID for manual text input. To enable PDF parsing:

```bash
# Install and run GROBID (see documentation)
# Or start backend with --no-grobid flag
cd backend
python app.py --no-grobid
```

---

## 📚 Documentation

- **[RAG/README.md](RAG/README.md)**: RAG system documentation
- **[RAG/EVALUATION_REPORT.md](RAG/EVALUATION_REPORT.md)**: Performance analysis and comparison
- **[RAG/PIPELINE.md](RAG/PIPELINE.md)**: Detailed pipeline documentation
- **[fine_tuning/README.md](fine_tuning/README.md)**: Model training guide

---

## 📊 Dataset

- **Training Data**: 862 classes from OpenAlex taxonomy
- **Test Set**: 300 manually annotated articles
- **Taxonomy**: 1,449 hierarchical paths (7 root categories)

---

## 🙏 Acknowledgments

- **SciBERT**: Allen Institute for AI
- **ChromaDB**: Vector database for embeddings
- **GROBID**: PDF parsing
- **OpenAlex**: Research taxonomy and metadata


---

**Ready to classify research articles? Start the platform and upload your first PDF!** 🚀
