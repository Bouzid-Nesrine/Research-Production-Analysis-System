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
LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# RAG Configuration
RAG_CONFIG = {
    # Retrieval parameters
    "top_k": 10,  # Number of paths to retrieve (5-15 recommended)
    "similarity_threshold": 0.7,  # Minimum similarity score (0-1)
    "diversity_weight": 0.2,  # Weight for diversity in retrieval (0-1)
    
    # LLM parameters
    "temperature": 0.3,  # Lower = more deterministic (0.1-0.7)
    "max_new_tokens": 256,  # Maximum tokens for LLM response
    "top_p": 0.9,  # Nucleus sampling
    "do_sample": True,
    
    # Processing
    "batch_size": 8,  # For batch processing
    "max_retries": 3,  # Retry attempts on failure
    
    # ChromaDB
    "collection_name": "taxonomy_paths",
    "embedding_function": None,  # Will be set at runtime
}

# LLM Loading Configuration
LLM_LOAD_CONFIG = {
    "torch_dtype": "auto",
    "device_map": "auto",
    "load_in_8bit": False,  # Set to True if GPU memory is limited
    "trust_remote_code": True,
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
