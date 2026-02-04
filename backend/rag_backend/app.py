"""
Flask API Server for Research Paper Classification
Provides endpoints for PDF upload, extraction, and classification
"""

import os
import sys
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from functools import lru_cache

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add project root to path so RAG can be imported as a package
# Path: backend/rag_backend/app.py -> go up 2 levels to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"RAG folder exists: {(PROJECT_ROOT / 'RAG').exists()}")

from pdf_extractor import PDFExtractor, GrobidManager

# Import RAG components
try:
    from RAG.rag_pipeline import RAGClassificationPipeline
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG pipeline not available: {e}")
    RAG_AVAILABLE = False
    # Define a dummy class for type hints
    class RAGClassificationPipeline:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"])

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size

UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Simple in-memory cache for classification results
classification_cache: Dict[str, Dict] = {}

# Global instances
pdf_extractor: Optional[PDFExtractor] = None
rag_pipeline: Optional[RAGClassificationPipeline] = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_services():
    """Initialize PDF extractor and RAG pipeline"""
    global pdf_extractor, rag_pipeline
    
    logger.info("Initializing services...")
    
    # Initialize PDF extractor
    pdf_extractor = PDFExtractor(auto_start_grobid=True)
    
    # Initialize RAG pipeline if available
    if RAG_AVAILABLE:
        try:
            logger.info("Setting up RAG classification pipeline...")
            rag_pipeline = RAGClassificationPipeline(auto_setup=True)
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            rag_pipeline = None
    else:
        logger.warning("RAG pipeline not available")


def convert_path_to_hierarchy(path_string: str, confidence: float) -> list:
    """
    Convert classification path string to hierarchical structure for frontend
    
    Args:
        path_string: Classification path like "Domain > Field > Subfield > Topic"
        confidence: Overall confidence score
        
    Returns:
        List of ClassificationNode objects for frontend
    """
    if not path_string:
        return []
    
    # Ensure confidence is a float
    try:
        confidence = float(confidence) if confidence else 0.8
    except (ValueError, TypeError):
        confidence = 0.8
    
    # Split path by common separators
    parts = []
    for sep in [' > ', ' -> ', ' / ', ' >> ']:
        if sep in path_string:
            parts = [p.strip() for p in path_string.split(sep)]
            break
    
    if not parts:
        parts = [path_string.strip()]
    
    # Build hierarchical structure
    if not parts:
        return []
    
    # Calculate confidence decay through levels
    confidence_decay = 0.03  # 3% decay per level
    
    # Build from leaf to root, then reverse
    def build_tree(parts: list, current_confidence: float, depth: int = 0) -> dict:
        if not parts:
            return None
        
        node = {
            "id": parts[0].lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""),
            "name": parts[0],
            "confidence": max(0.5, current_confidence - (depth * confidence_decay))
        }
        
        if len(parts) > 1:
            child = build_tree(parts[1:], current_confidence, depth + 1)
            if child:
                node["children"] = [child]
        
        return node
    
    root = build_tree(parts, confidence)
    return [root] if root else []


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "grobid": GrobidManager.is_grobid_running(),
            "rag_pipeline": rag_pipeline is not None
        }
    })


@app.route('/api/grobid/status', methods=['GET'])
def grobid_status():
    """Check GROBID status"""
    is_running = GrobidManager.is_grobid_running()
    return jsonify({
        "running": is_running,
        "url": pdf_extractor.grobid_url if pdf_extractor else None
    })


@app.route('/api/grobid/start', methods=['POST'])
def start_grobid():
    """Start GROBID container"""
    try:
        success = GrobidManager.start_grobid_container()
        if success:
            return jsonify({"status": "started", "message": "GROBID is now running"})
        else:
            return jsonify({"status": "error", "message": "Failed to start GROBID"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/extract', methods=['POST'])
def extract_pdf():
    """
    Extract title and abstract from uploaded PDF
    
    Request: multipart/form-data with 'file' field
    Response: JSON with extracted metadata
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400
    
    try:
        start_time = time.time()
        
        # Read file bytes
        pdf_bytes = file.read()
        filename = secure_filename(file.filename)
        
        # Extract content using GROBID
        if pdf_extractor is None:
            return jsonify({"error": "PDF extractor not initialized"}), 500
        
        extraction_result = pdf_extractor.extract_from_pdf_bytes(pdf_bytes, filename)
        
        processing_time = time.time() - start_time
        
        if extraction_result.get('error'):
            return jsonify({
                "error": extraction_result['error'],
                "processing_time": processing_time
            }), 500
        
        return jsonify({
            "success": True,
            "data": {
                "title": extraction_result.get('title'),
                "abstract": extraction_result.get('abstract'),
                "authors": extraction_result.get('authors', []),
                "year": extraction_result.get('year'),
                "journal": extraction_result.get('journal'),
                "doi": extraction_result.get('doi'),
                "keywords": extraction_result.get('keywords', [])
            },
            "processing_time": processing_time
        })
        
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/classify', methods=['POST'])
def classify_article():
    """
    Classify article using RAG pipeline with caching
    
    Request: JSON with 'title' and 'abstract' fields
    Response: JSON with classification result
    """
    if rag_pipeline is None:
        return jsonify({"error": "RAG pipeline not available"}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    title = data.get('title')
    abstract = data.get('abstract')
    
    if not title or not abstract:
        return jsonify({"error": "Both 'title' and 'abstract' are required"}), 400
    
    try:
        start_time = time.time()
        
        # Check cache first
        cache_key = hashlib.md5(f"{title}:{abstract[:200]}".encode()).hexdigest()
        if cache_key in classification_cache:
            cached = classification_cache[cache_key]
            cached['from_cache'] = True
            cached['processing_time'] = 0.01
            logger.info(f"Returning cached classification for: {title[:50]}...")
            return jsonify(cached)
        
        # Classify using RAG pipeline
        result = rag_pipeline.classify_article(
            title=title,
            abstract=abstract,
            return_metadata=True
        )
        
        processing_time = time.time() - start_time
        
        if 'error' in result:
            return jsonify({
                "error": result['error'],
                "processing_time": processing_time
            }), 500
        
        classification = result.get('classification', {})
        metadata = result.get('metadata', {})
        
        response_data = {
            "success": True,
            "classification": {
                "path": classification.get('path'),
                "confidence": classification.get('confidence'),
                "reasoning": classification.get('reasoning'),
                "valid": classification.get('valid')
            },
            "metadata": {
                "retrieval_time": metadata.get('retrieval', {}).get('retrieval_time'),
                "classification_time": metadata.get('classification', {}).get('classification_time'),
                "total_time": metadata.get('total_time'),
                "model": metadata.get('classification', {}).get('model')
            },
            "processing_time": processing_time
        }
        
        # Cache the result
        classification_cache[cache_key] = response_data
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_and_classify():
    """
    Upload PDF, extract content, and classify in one request
    
    Request: multipart/form-data with 'file' field
    Response: JSON with extraction and classification results
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400
    
    try:
        total_start_time = time.time()
        
        # Read file bytes
        pdf_bytes = file.read()
        filename = secure_filename(file.filename)
        
        # Step 1: Extract content using GROBID
        extraction_start = time.time()
        
        if pdf_extractor is None:
            return jsonify({"error": "PDF extractor not initialized"}), 500
        
        extraction_result = pdf_extractor.extract_from_pdf_bytes(pdf_bytes, filename)
        extraction_time = time.time() - extraction_start
        
        if extraction_result.get('error'):
            return jsonify({
                "error": f"Extraction failed: {extraction_result['error']}",
                "processing_time": time.time() - total_start_time
            }), 500
        
        title = extraction_result.get('title')
        abstract = extraction_result.get('abstract')
        
        if not title or not abstract:
            return jsonify({
                "error": "Could not extract title and/or abstract from PDF",
                "extraction": extraction_result,
                "processing_time": time.time() - total_start_time
            }), 400
        
        # Step 2: Classify using RAG pipeline
        classification_result = None
        classification_time = 0
        
        if rag_pipeline is not None:
            classification_start = time.time()
            
            rag_result = rag_pipeline.classify_article(
                title=title,
                abstract=abstract,
                return_metadata=True
            )
            classification_time = time.time() - classification_start
            
            if 'error' not in rag_result:
                classification_result = rag_result.get('classification', {})
        
        total_time = time.time() - total_start_time
        
        # Build response
        response = {
            "success": True,
            "articleInfo": {
                "title": title,
                "authors": extraction_result.get('authors', []),
                "year": extraction_result.get('year'),
                "journal": extraction_result.get('journal'),
                "doi": extraction_result.get('doi')
            },
            "abstract": abstract,
            "processingTime": total_time,
            "extractionTime": extraction_time,
            "classificationTime": classification_time
        }
        
        if classification_result:
            path_string = classification_result.get('path', '')
            confidence = classification_result.get('confidence', 0.8)
            
            response["path"] = convert_path_to_hierarchy(path_string, confidence)
            response["confidence"] = confidence
            response["reasoning"] = classification_result.get('reasoning')
            response["model"] = rag_pipeline.llm_model_name if rag_pipeline else "unknown"
        else:
            # No classification available - return extraction only
            response["path"] = []
            response["confidence"] = 0
            response["model"] = "none"
            response["warning"] = "Classification not available - RAG pipeline not initialized"
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Upload and classify error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/pipeline/status', methods=['GET'])
def pipeline_status():
    """Get status of the classification pipeline"""
    return jsonify({
        "rag_available": rag_pipeline is not None,
        "grobid_running": GrobidManager.is_grobid_running(),
        "pdf_extractor_ready": pdf_extractor is not None,
        "statistics": rag_pipeline.get_statistics() if rag_pipeline else None
    })


@app.route('/api/pipeline/reset', methods=['POST'])
def reset_pipeline():
    """Reset and reinitialize the RAG pipeline"""
    global rag_pipeline
    
    if not RAG_AVAILABLE:
        return jsonify({"error": "RAG pipeline module not available"}), 503
    
    try:
        rag_pipeline = RAGClassificationPipeline(auto_setup=True)
        return jsonify({"status": "success", "message": "Pipeline reset successfully"})
    except Exception as e:
        logger.error(f"Failed to reset pipeline: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File is too large. Maximum size is 10MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({"error": "Internal server error"}), 500


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Research Paper Classification API')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-grobid', action='store_true', help='Skip GROBID initialization')
    
    args = parser.parse_args()
    
    # Initialize services
    initialize_services()
    
    # Check GROBID status
    if not args.no_grobid:
        if GrobidManager.is_grobid_running():
            logger.info("GROBID is running")
        else:
            logger.warning("GROBID is not running. Starting GROBID container...")
            if GrobidManager.start_grobid_container():
                logger.info("GROBID started successfully")
            else:
                logger.warning("Failed to start GROBID. PDF extraction will not work.")
    
    # Start server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
