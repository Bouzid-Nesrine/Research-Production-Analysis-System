"""
Configuration for RAG-based Taxonomy Classification System
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TAXONOMY_PATH = PROJECT_ROOT / "Taxonomy Building" / "final_alex_taxonomy.json"
CHROMA_DB_PATH = PROJECT_ROOT / "RAG" / "chroma_db"
LOGS_PATH = PROJECT_ROOT / "RAG" / "logs"

# Ensure directories exist
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# Model Configuration
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # Options: all-MiniLM-L6-v2, all-mpnet-base-v2, allenai-specter
LLM_MODEL_NAME = "gemini-2.5-flash-lite"  # Options: gemini-2.5-flash-lite, gemini-2.0-flash, gemini-1.5-flash

# API Configuration (for Google AI Studio)
# Set GOOGLE_API_KEY in .env file
GOOGLE_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# RAG Configuration
RAG_CONFIG = {
    # Retrieval parameters
    "top_k": 5,  # Number of best paths to retrieve (no threshold filtering)
    "similarity_threshold": 0.0,  # Disabled - always return top_k best matches
    "diversity_weight": 0.2,  # Weight for diversity in retrieval (0-1)
    
    # LLM parameters (for Alibaba Cloud API)
    "temperature": 0.3,  # Lower = more deterministic (0-2)
    "max_tokens": 256,  # Maximum tokens for LLM response
    "top_p": 0.9,  # Nucleus sampling (0-1)
    
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
    "timeout": 60,  # Request timeout in seconds
    "max_retries": 3,  # Number of retry attempts
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
