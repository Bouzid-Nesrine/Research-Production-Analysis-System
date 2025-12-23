"""
Setup Script - Initialize RAG classification pipeline
"""

import sys
from pathlib import Path
import argparse
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config import TAXONOMY_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL_NAME
from taxonomy_parser import TaxonomyParser
from vector_db_manager import VectorDBManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_database(reset: bool = False):
    """
    Setup ChromaDB with taxonomy paths
    
    Args:
        reset: Reset existing database
    """
    logger.info("=" * 60)
    logger.info("RAG Classification Pipeline Setup")
    logger.info("=" * 60)
    
    # Step 1: Parse taxonomy
    logger.info("\n[1/3] Parsing taxonomy...")
    parser = TaxonomyParser(TAXONOMY_PATH)
    paths = parser.extract_all_paths()
    
    stats = parser.get_statistics()
    logger.info(f"✓ Extracted {stats['total_paths']} paths")
    logger.info(f"  - Max level: {stats['max_level']}")
    logger.info(f"  - Domains: {len(stats['domains'])}")
    
    # Save paths for reference
    paths_output = Path(__file__).parent / 'taxonomy_paths.json'
    parser.save_paths(paths_output)
    
    # Step 2: Initialize ChromaDB
    logger.info(f"\n[2/3] Initializing ChromaDB...")
    logger.info(f"  - Database path: {CHROMA_DB_PATH}")
    logger.info(f"  - Embedding model: {EMBEDDING_MODEL_NAME}")
    
    db_manager = VectorDBManager(
        db_path=CHROMA_DB_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    
    db_manager.initialize_collection(reset=reset)
    
    # Step 3: Populate database
    logger.info(f"\n[3/3] Populating vector database...")
    db_manager.populate_from_paths(paths, show_progress=True)
    
    # Verify
    collection_stats = db_manager.get_collection_stats()
    logger.info(f"\n✓ Database setup complete!")
    logger.info(f"  - Total paths in DB: {collection_stats['total_paths']}")
    
    # Test retrieval
    logger.info(f"\n[Test] Testing retrieval...")
    test_query = "machine learning deep neural networks"
    results = db_manager.retrieve_relevant_paths(test_query, top_k=3)
    
    logger.info(f"\nTop 3 matches for '{test_query}':")
    for path_info in results['retrieved_paths']:
        logger.info(f"  {path_info['rank']}. {path_info['path']}")
        logger.info(f"     Similarity: {path_info['similarity']:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Setup Complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Run: python rag_pipeline.py")
    logger.info("2. Or use the Jupyter notebook for interactive testing")
    logger.info("=" * 60)


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Setup RAG Classification Pipeline"
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset existing database'
    )
    
    args = parser.parse_args()
    
    try:
        setup_database(reset=args.reset)
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
