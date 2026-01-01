"""
Vector Database Manager - Initialize and manage ChromaDB for taxonomy paths
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class VectorDBManager:
    """Manage ChromaDB for taxonomy path storage and retrieval"""
    
    def __init__(
        self,
        db_path: str,
        collection_name: str = "taxonomy_paths",
        embedding_model_name: str = "all-mpnet-base-v2"
    ):
        """
        Initialize ChromaDB manager
        
        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the collection
            embedding_model_name: Sentence transformer model name
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Get or create collection
        self.collection = None
        
    def initialize_collection(self, reset: bool = False):
        """
        Initialize or reset the collection
        
        Args:
            reset: If True, delete existing collection and create new
        """
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception as e:
                logger.debug(f"No existing collection to delete: {e}")
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Research taxonomy classification paths",
                "hnsw:space": "cosine"  # Use cosine distance for similarity search
            }
        )
        
        logger.info(f"Initialized collection: {self.collection_name}")
    
    def populate_from_paths(
        self,
        paths: List[Dict[str, Any]],
        batch_size: int = 100,
        show_progress: bool = True
    ):
        """
        Populate ChromaDB with taxonomy paths
        
        Args:
            paths: List of path dictionaries from TaxonomyParser
            batch_size: Number of paths to process at once
            show_progress: Show progress bar
        """
        if self.collection is None:
            self.initialize_collection()
        
        logger.info(f"Populating ChromaDB with {len(paths)} paths")
        
        # Process in batches
        iterator = range(0, len(paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding paths")
        
        for i in iterator:
            batch = paths[i:i + batch_size]
            
            # Prepare data
            ids = [p['id'] for p in batch]
            documents = [p['description'] for p in batch]
            metadatas = [
                {
                    'path': p['full_path'],
                    'domain': p['domain'],
                    'field': p['field'],
                    'level': p['level'],
                    'keywords': ','.join(p['keywords']),
                }
                for p in batch
            ]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        
        total_count = self.collection.count()
        logger.info(f"Successfully added {total_count} paths to ChromaDB")
    
    def retrieve_relevant_paths(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        filter_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve most relevant taxonomy paths for a query
        
        Args:
            query_text: Article title + abstract
            top_k: Number of paths to retrieve
            similarity_threshold: Minimum similarity score (0-1)
            filter_domain: Optional domain filter
            
        Returns:
            Dictionary with retrieved paths and metadata
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query_text,
            convert_to_numpy=True
        ).tolist()
        
        # Prepare query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ['documents', 'metadatas', 'distances']
        }
        
        # Add domain filter if specified
        if filter_domain:
            query_params["where"] = {"domain": filter_domain}
        
        # Query ChromaDB
        results = self.collection.query(**query_params)
        
        # Process results
        paths = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Convert distance to similarity (assuming cosine distance)
            similarity = 1 - distance
            
            # Apply threshold if specified
            if similarity_threshold and similarity < similarity_threshold:
                continue
            
            paths.append({
                'rank': i + 1,
                'path': metadata['path'],
                'domain': metadata['domain'],
                'field': metadata['field'],
                'level': metadata['level'],
                'description': doc,
                'similarity': similarity,
                'distance': distance
            })
        
        return {
            'query': query_text,
            'retrieved_paths': paths,
            'total_retrieved': len(paths),
            'top_k': top_k
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        if self.collection is None:
            return {"error": "Collection not initialized"}
        
        count = self.collection.count()
        
        # Sample some items to get metadata
        if count > 0:
            sample = self.collection.peek(limit=min(10, count))
            domains = set()
            levels = set()
            
            for metadata in sample['metadatas']:
                domains.add(metadata.get('domain', 'Unknown'))
                levels.add(metadata.get('level', 0))
            
            return {
                "total_paths": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "sample_domains": list(domains),
                "sample_levels": list(levels),
            }
        
        return {
            "total_paths": 0,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
        }
    
    def search_by_keywords(
        self,
        keywords: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search paths by keywords
        
        Args:
            keywords: List of keywords to search
            top_k: Number of results
            
        Returns:
            List of matching paths
        """
        # Combine keywords into query
        query = " ".join(keywords)
        result = self.retrieve_relevant_paths(query, top_k)
        return result['retrieved_paths']
    
    def get_paths_by_domain(self, domain: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all paths in a specific domain
        
        Args:
            domain: Domain name
            limit: Maximum number of results
            
        Returns:
            List of paths in domain
        """
        if self.collection is None:
            raise ValueError("Collection not initialized")
        
        results = self.collection.get(
            where={"domain": domain},
            limit=limit,
            include=['metadatas', 'documents']
        )
        
        paths = []
        for metadata, doc in zip(results['metadatas'], results['documents']):
            paths.append({
                'path': metadata['path'],
                'domain': metadata['domain'],
                'description': doc,
            })
        
        return paths


def main():
    """Example usage and testing"""
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    from config import CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, TAXONOMY_PATH
    from taxonomy_parser import TaxonomyParser
    
    # Initialize database manager
    db_manager = VectorDBManager(
        db_path=CHROMA_DB_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    
    # Parse taxonomy
    print("\n=== Parsing Taxonomy ===")
    parser = TaxonomyParser(TAXONOMY_PATH)
    paths = parser.extract_all_paths()
    print(f"Extracted {len(paths)} paths")
    
    # Initialize and populate collection
    print("\n=== Initializing ChromaDB ===")
    db_manager.initialize_collection(reset=True)
    db_manager.populate_from_paths(paths)
    
    # Get statistics
    print("\n=== Collection Statistics ===")
    stats = db_manager.get_collection_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test retrieval
    print("\n=== Testing Retrieval ===")
    test_query = """
    Title: Deep Learning for Image Classification
    Abstract: This paper presents a novel deep learning approach for image 
    classification using convolutional neural networks. We demonstrate 
    state-of-the-art performance on benchmark datasets.
    """
    
    results = db_manager.retrieve_relevant_paths(test_query, top_k=5)
    print(f"\nTop {results['top_k']} relevant paths:")
    for path_info in results['retrieved_paths']:
        print(f"\n{path_info['rank']}. {path_info['path']}")
        print(f"   Similarity: {path_info['similarity']:.4f}")
        print(f"   Domain: {path_info['domain']}")


if __name__ == "__main__":
    main()
