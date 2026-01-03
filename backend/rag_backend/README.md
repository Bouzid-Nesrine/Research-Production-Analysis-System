# Research Paper Classification Backend

This backend provides an API for uploading research paper PDFs, extracting title and abstract using GROBID, and classifying them using the RAG pipeline.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Frontend       │────▶│  Flask API      │────▶│  RAG Pipeline   │
│  (React)        │     │  (Port 5000)    │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │                 │
                        │  GROBID         │
                        │  (Docker:8070)  │
                        │                 │
                        └─────────────────┘
```

## Prerequisites

1. **Docker** - Required for running GROBID
2. **Python 3.8+** - Required for the Flask API
3. **Google API Key** - Required for the RAG pipeline (set in `.env`)

## Quick Start

### 1. Start the Backend

```bash
cd backend
chmod +x start.sh
./start.sh
```

This will:
- Check and start Docker
- Start GROBID container (if not running)
- Install Python dependencies
- Start the Flask API server on port 5000

### 2. Start the Frontend

```bash
cd nlp-platform-ui
npm install
npm run dev
```

Frontend will be available at `http://localhost:5173`

## Manual Setup

### Start GROBID Manually

```bash
docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0
```

### Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Configure Environment

Copy the `.env` file from RAG folder or create one:

```bash
cp ../RAG/.env .env
```

Make sure it contains:
```
GOOGLE_API_KEY=your_api_key_here
```

### Start the Server

```bash
python app.py --host 127.0.0.1 --port 5000
```

## API Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-12-25T10:00:00",
  "services": {
    "grobid": true,
    "rag_pipeline": true
  }
}
```

### Upload and Classify (Main Endpoint)
```http
POST /api/upload
Content-Type: multipart/form-data

file: <PDF file>
```

Response:
```json
{
  "success": true,
  "articleInfo": {
    "title": "Article Title",
    "authors": ["Author 1", "Author 2"],
    "year": 2024,
    "journal": "Journal Name",
    "doi": "10.1000/xyz"
  },
  "abstract": "Article abstract text...",
  "path": [
    {
      "id": "domain",
      "name": "Computer Science",
      "confidence": 0.95,
      "children": [...]
    }
  ],
  "confidence": 0.85,
  "processingTime": 5.2,
  "model": "gemini-2.5-flash-lite"
}
```

### Extract Only
```http
POST /api/extract
Content-Type: multipart/form-data

file: <PDF file>
```

### Classify Text
```http
POST /api/classify
Content-Type: application/json

{
  "title": "Article Title",
  "abstract": "Article abstract..."
}
```

### GROBID Status
```http
GET /api/grobid/status
```

### Start GROBID
```http
POST /api/grobid/start
```

### Pipeline Status
```http
GET /api/pipeline/status
```

## File Structure

```
backend/
├── app.py              # Flask API server
├── pdf_extractor.py    # GROBID integration for PDF extraction
├── requirements.txt    # Python dependencies
├── start.sh           # Startup script
├── README.md          # This file
└── uploads/           # Temporary upload folder (auto-created)
```

## Troubleshooting

### GROBID Not Starting

1. Check if Docker is running: `docker info`
2. Check if port 8070 is available: `lsof -i :8070`
3. View GROBID logs: `docker logs grobid`

### RAG Pipeline Not Available

1. Ensure `.env` file exists with `GOOGLE_API_KEY`
2. Check if ChromaDB is initialized: Look for `RAG/chroma_db/` folder
3. Run RAG setup: `cd ../RAG && python setup_pipeline.py`

### CORS Errors in Frontend

The backend is configured to allow requests from:
- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `http://localhost:3000`

If using a different port, update `CORS()` in `app.py`.

## Development

### Running in Debug Mode

```bash
python app.py --debug
```

### Testing Extraction Only

```bash
curl -X POST -F "file=@paper.pdf" http://127.0.0.1:5000/api/extract
```

### Testing Classification Only

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"title": "Test Title", "abstract": "Test abstract..."}' \
  http://127.0.0.1:5000/api/classify
```
