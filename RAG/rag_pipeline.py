"""
RAG Classification Pipeline - Complete end-to-end classification system
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging
from datetime import datetime
from functools import lru_cache

from taxonomy_parser import TaxonomyParser
from vector_db_manager import VectorDBManager
from RAG.llm_classifier_api import LLMClassifier
from config import (
    TAXONOMY_PATH,
    CHROMA_DB_PATH,
    RAG_CONFIG,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_API_CONFIG,
    LOGS_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_PATH / f'rag_pipeline_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class RAGClassificationPipeline:
    """Complete RAG-based taxonomy classification pipeline"""
    
    def __init__(
        self,
        taxonomy_path: Optional[str] = None,
        db_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        config: Optional[Dict] = None,
        auto_setup: bool = False
    ):
        """
        Initialize RAG classification pipeline
        
        Args:
            taxonomy_path: Path to taxonomy JSON
            db_path: Path to ChromaDB storage
            embedding_model: Embedding model name
            llm_model: LLM model name
            config: Configuration dictionary
            auto_setup: Automatically setup database on init
        """
        # Use defaults from config if not provided
        self.taxonomy_path = taxonomy_path or TAXONOMY_PATH
        self.db_path = db_path or CHROMA_DB_PATH
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL_NAME
        self.llm_model_name = llm_model or LLM_MODEL_NAME
        self.config = config or RAG_CONFIG
        
        # Initialize components
        self.taxonomy_parser = None
        self.taxonomy_paths = None
        self.db_manager = None
        self.llm_classifier = None
        
        # Statistics
        self.stats = {
            'total_classified': 0,
            'successful': 0,
            'failed': 0,
            'avg_retrieval_time': 0,
            'avg_classification_time': 0,
        }
        
        if auto_setup:
            self.setup()
    
    def setup(self, reset_db: bool = False):
        """
        Setup all pipeline components
        
        Args:
            reset_db: Reset ChromaDB collection
        """
        logger.info("Setting up RAG classification pipeline...")
        
        # 1. Parse taxonomy
        logger.info("Parsing taxonomy...")
        self.taxonomy_parser = TaxonomyParser(self.taxonomy_path)
        self.taxonomy_paths = self.taxonomy_parser.extract_all_paths()
        
        stats = self.taxonomy_parser.get_statistics()
        logger.info(f"Loaded {stats['total_paths']} taxonomy paths")
        
        # 2. Initialize vector database
        logger.info("Initializing vector database...")
        self.db_manager = VectorDBManager(
            db_path=self.db_path,
            embedding_model_name=self.embedding_model_name
        )
        
        self.db_manager.initialize_collection(reset=reset_db)
        
        # 3. Populate database if empty or reset
        current_count = self.db_manager.collection.count()
        if current_count == 0 or reset_db:
            logger.info("Populating vector database...")
            self.db_manager.populate_from_paths(self.taxonomy_paths)
        else:
            logger.info(f"Using existing database with {current_count} paths")
        
        # 4. Initialize LLM (lazy loading - only when needed)
        logger.info("Pipeline setup complete (LLM will be loaded on first use)")
    
    def _ensure_llm_loaded(self):
        """Lazy load LLM classifier"""
        if self.llm_classifier is None:
            logger.info("Initializing LLM classifier with Alibaba Cloud API...")
            self.llm_classifier = LLMClassifier(
                model_name=self.llm_model_name,
                api_base_url=LLM_API_CONFIG.get('api_base_url')
            )
    
    def classify_article(
        self,
        title: str,
        abstract: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Classify a single article
        
        Args:
            title: Article title
            abstract: Article abstract
            top_k: Number of paths to retrieve (uses config default if None)
            temperature: LLM temperature (uses config default if None)
            return_metadata: Include retrieval and timing metadata
            
        Returns:
            Classification result dictionary
        """
        import time
        
        if self.db_manager is None:
            raise ValueError("Pipeline not setup. Call setup() first.")
        
        # Use config defaults if not specified
        top_k = top_k or self.config['top_k']
        temperature = temperature or self.config['temperature']
        
        result = {
            'article': {
                'title': title,
                'abstract': abstract
            },
            'classification': None,
            'metadata': {} if return_metadata else None
        }
        
        try:
            # Step 1: Retrieve relevant paths
            retrieval_start = time.time()
            
            article_text = f"Title: {title}\n\nAbstract: {abstract}"
            retrieval_result = self.db_manager.retrieve_relevant_paths(
                query_text=article_text,
                top_k=top_k,
                similarity_threshold=self.config.get('similarity_threshold')
            )
            
            retrieval_time = time.time() - retrieval_start
            
            if not retrieval_result['retrieved_paths']:
                raise ValueError("No relevant paths retrieved")
            
            relevant_paths = [
                p['path'] for p in retrieval_result['retrieved_paths']
            ]
            
            # Step 2: Classify with LLM
            self._ensure_llm_loaded()
            
            classification_start = time.time()
            
            llm_result = self.llm_classifier.classify_article(
                title=title,
                abstract=abstract,
                relevant_paths=relevant_paths,
                temperature=temperature,
                max_tokens=self.config.get('max_tokens', 256),
                top_p=self.config['top_p']
            )
            
            classification_time = time.time() - classification_start
            
            classified_path = llm_result['classification']['path']
            is_valid = self._validate_path(classified_path, relevant_paths)
            
            result['classification'] = {
                'path': classified_path,
                'confidence': llm_result['classification']['confidence'],
                'reasoning': llm_result['classification']['reasoning'],
                'valid': is_valid
            }
            
            if return_metadata:
                result['metadata'] = {
                    'retrieval': {
                        'top_k': top_k,
                        'paths_retrieved': len(relevant_paths),
                        'retrieval_time': retrieval_time,
                        'similarity_scores': [
                            p['similarity'] for p in retrieval_result['retrieved_paths']
                        ]
                    },
                    'classification': {
                        'model': self.llm_model_name,
                        'temperature': temperature,
                        'classification_time': classification_time,
                        'prompt_length': llm_result['prompt_length'],
                        'response_length': llm_result['response_length']
                    },
                    'total_time': retrieval_time + classification_time
                }
            
            self.stats['successful'] += 1
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            result['error'] = str(e)
            self.stats['failed'] += 1
        
        finally:
            self.stats['total_classified'] += 1
        
        return result
    
    def _validate_path(self, classified_path: str, retrieved_paths: List[str]) -> bool:
        """
        Validate that classified path exists in retrieved paths
        
        Args:
            classified_path: Path returned by LLM
            retrieved_paths: List of retrieved paths
            
        Returns:
            True if path is valid
        """
        if not classified_path:
            return False
        
        # Exact match
        if classified_path in retrieved_paths:
            return True
        
        # Fuzzy match (handle minor formatting differences)
        classified_normalized = classified_path.lower().strip()
        for path in retrieved_paths:
            if classified_normalized == path.lower().strip():
                return True
        
        # Check if it's a substring or superstring
        for path in retrieved_paths:
            if classified_normalized in path.lower() or path.lower() in classified_normalized:
                logger.warning(f"Fuzzy match: '{classified_path}' ~ '{path}'")
                return True
        
        logger.warning(f"Invalid path: '{classified_path}' not in retrieved paths")
        return False
    
    def batch_classify(
        self,
        articles: List[Dict[str, str]],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple articles
        
        Args:
            articles: List of dicts with 'title' and 'abstract' keys
            batch_size: Processing batch size (currently processes sequentially)
            show_progress: Show progress bar
            
        Returns:
            List of classification results
        """
        from tqdm import tqdm
        
        results = []
        
        iterator = articles
        if show_progress:
            iterator = tqdm(articles, desc="Classifying articles")
        
        for article in iterator:
            result = self.classify_article(
                title=article['title'],
                abstract=article['abstract']
            )
            results.append(result)
        
        return results
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        format: str = 'json'
    ):
        """
        Save classification results
        
        Args:
            results: List of classification results
            output_path: Path to save results
            format: Output format ('json', 'jsonl', 'csv')
        """
        output_path = Path(output_path)
        
        if format == 'json':
            # Filter out None results
            valid_results = [r for r in results if r is not None]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(valid_results, f, indent=2, ensure_ascii=False)
        
        elif format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        elif format == 'csv':
            import pandas as pd
            
            # Flatten results for CSV
            flat_results = []
            for r in results:
                # Handle None or failed results
                if r is None:
                    flat = {
                        'title': '',
                        'abstract': '',
                        'classified_path': '',
                        'confidence': '',
                        'valid': False,
                        'error': 'No result'
                    }
                elif 'error' in r:
                    flat = {
                        'title': r.get('article', {}).get('title', '') if 'article' in r else '',
                        'abstract': r.get('article', {}).get('abstract', '') if 'article' in r else '',
                        'classified_path': '',
                        'confidence': '',
                        'valid': False,
                        'error': r.get('error', 'Unknown error')
                    }
                else:
                    flat = {
                        'title': r.get('article', {}).get('title', ''),
                        'abstract': r.get('article', {}).get('abstract', ''),
                        'classified_path': r.get('classification', {}).get('path', ''),
                        'confidence': r.get('classification', {}).get('confidence', ''),
                        'valid': r.get('classification', {}).get('valid', False),
                        'error': ''
                    }
                flat_results.append(flat)
            
            df = pd.DataFrame(flat_results)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            **self.stats,
            'success_rate': self.stats['successful'] / max(self.stats['total_classified'], 1),
            'database_stats': self.db_manager.get_collection_stats() if self.db_manager else {}
        }


def main():
    """Example usage"""
    # Initialize pipeline
    pipeline = RAGClassificationPipeline(auto_setup=True)
    
    # Example articles
    articles = [
        {
            'title': 'Deep Learning for Medical Image Segmentation',
            'abstract': 'This paper presents a novel deep learning approach for automated '
                       'medical image segmentation using convolutional neural networks.'
        },
        {
            'title': 'Climate Change Impact on Agricultural Productivity',
            'abstract': 'We analyze the effects of climate change on crop yields across '
                       'different regions using statistical models and satellite data.'
        },
        {
            'title': 'Quantum Computing Algorithms for Optimization',
            'abstract': 'This work introduces new quantum algorithms for solving complex '
                       'optimization problems with applications in logistics and finance.'
        }
    ]
    
    # Classify articles
    print("\n=== Classifying Articles ===\n")
    results = pipeline.batch_classify(articles, show_progress=True)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n--- Article {i} ---")
        print(f"Title: {result['article']['title']}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Path: {result['classification']['path']}")
            print(f"Confidence: {result['classification']['confidence']}")
            print(f"Valid: {result['classification']['valid']}")
            print(f"Time: {result['metadata']['total_time']:.2f}s")
    
    # Save results
    output_path = Path(__file__).parent / 'results' / 'example_results.json'
    output_path.parent.mkdir(exist_ok=True)
    pipeline.save_results(results, output_path)
    
    # Statistics
    print("\n=== Pipeline Statistics ===")
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        if key != 'database_stats':
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
