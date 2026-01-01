"""
Configuration for RAG-based Taxonomy Classification System
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TAXONOMY_PATH = PROJECT_ROOT / "Taxonomy Building" / "preprocessed_taxonomy.json"
CHROMA_DB_PATH = PROJECT_ROOT / "RAG" / "chroma_db"
LOGS_PATH = PROJECT_ROOT / "RAG" / "logs"

# Ensure directories exist
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Model Configuration
# Using lighter model for faster embeddings (all-MiniLM-L6-v2 is 5x faster than mpnet)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (accurate), allenai-specter
LLM_MODEL_NAME = "gemini-2.5-flash-lite"  # Options: gemini-2.0-flash-exp (fast, stable), gemini-1.5-flash, gemini-1.5-pro

# API Configuration (for Google AI Studio)
# Set GOOGLE_API_KEY in .env file
GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# RAG Configuration
RAG_CONFIG = {
    # Retrieval parameters (optimized for speed)
    "top_k": 5,  # Reduced from 5 - fewer paths = faster LLM processing
    "similarity_threshold": 0.0,  # Disabled - always return top_k best matches
    "diversity_weight": 0.1,  # Reduced for faster retrieval
    
    # LLM parameters (optimized for speed)
    "temperature": 0.1,  # Lower = faster, more deterministic
    "max_tokens": 150,  # Reduced from 256 - we only need path + confidence
    "top_p": 0.8,  # Slightly reduced for faster generation
    
    # Processing
    "batch_size": 8,  # For batch processing
    "max_retries": 3,  # Retry attempts on failure
    
    # ChromaDB
    "collection_name": "taxonomy_paths",
    "embedding_function": None,  # Will be set at runtime
}

# LLM API Configuration (for Google AI Studio)
LLM_API_CONFIG = {
    "model_name": LLM_MODEL_NAME,
    "api_base_url": GOOGLE_API_BASE_URL,
    "timeout": 30,  # Reduced timeout for faster failure detection
    "max_retries": 1,  # Reduced retries for speed
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": LOGS_PATH / "rag_classification.log",
}

# Evaluation Configuration
EVAL_CONFIG = {
    "metrics": ["exact_match", "hierarchical_accuracy", "domain_accuracy"],
    "test_size": 0.2,
    "random_seed": 42,
}

# API Configuration (if deploying as service)
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
}

# Cache Configuration
CACHE_CONFIG = {
    "enable_embedding_cache": True,
    "cache_size": 1000,  # LRU cache size
    "enable_result_cache": True,
}
