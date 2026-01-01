"""
Pinecone Vector Database Manager (Alternative to ChromaDB)
Cloud-hosted vector database for taxonomy paths

Installation:
    pip install "pinecone[grpc]"
"""

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()


class PineconeVectorDB:
    """Manages taxonomy vectors in Pinecone cloud database"""
    
    def __init__(
        self,
        index_name: str = "taxonomy-paths",
        dimension: int = 384,
        metric: str = "cosine"
    ):
        """
        Initialize Pinecone connection
        
        Args:
            index_name: Name of Pinecone index
            dimension: Vector dimensions (384 for all-MiniLM-L6-v2)
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        # Get Pinecone API key
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in .env")
        
        # Initialize Pinecone client (new API)
        self.pc = Pinecone(api_key=api_key)
        
        self.index_name = index_name
        self.dimension = dimension
        
        # Create index if doesn't exist
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"Creating new index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        # Connect to index
        self.index = self.pc.Index(index_name)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def populate_from_taxonomy(self, taxonomy_paths: List[str]):
        """
        Upload taxonomy paths to Pinecone
        
        Args:
            taxonomy_paths: List of taxonomy paths to embed and upload
        """
        print(f"Embedding {len(taxonomy_paths)} taxonomy paths...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            taxonomy_paths,
            show_progress_bar=True
        )
        
        # Prepare vectors for upload
        vectors = []
        for i, (path, embedding) in enumerate(zip(taxonomy_paths, embeddings)):
            vectors.append({
                "id": f"path_{i}",
                "values": embedding.tolist(),
                "metadata": {"taxonomy_path": path}
            })
        
        # Upload in batches (Pinecone recommends batch size 100)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
        
        print(f"✓ Successfully uploaded {len(taxonomy_paths)} paths to Pinecone")
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for most similar taxonomy paths
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of (taxonomy_path, similarity_score) tuples
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract paths and scores
        matches = []
        for match in results['matches']:
            path = match['metadata']['taxonomy_path']
            score = match['score']
            matches.append((path, score))
        
        return matches
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.get('total_vector_count', 0),
            "dimension": stats.get('dimension', self.dimension),
            "index_fullness": stats.get('index_fullness', 0)
        }
    
    def delete_all(self):
        """Delete all vectors from index"""
        self.index.delete(delete_all=True)
        print("✓ All vectors deleted from Pinecone")


if __name__ == "__main__":
    """Example usage"""
    
    # Initialize
    db = PineconeVectorDB()
    
    # Load taxonomy paths - try multiple locations
    import os
    taxonomy_file = None
    possible_paths = [
        'preprocessed_taxonomy.json',
        '../Taxonomy Building/preprocessed_taxonomy.json',
        '../fine_tuning_new/preprocessed_taxonomy.json',
        '../fine_tuning/final_combined_taxonomy.json',
        '../Taxonomy Building/final_combined_taxonomy.json'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            taxonomy_file = path
            print(f"Found taxonomy file: {path}")
            break
    
    if not taxonomy_file:
        raise FileNotFoundError("No taxonomy file found. Please specify the correct path.")
    
    from taxonomy_parser import TaxonomyParser
    parser = TaxonomyParser(taxonomy_file)
    taxonomy_data = parser.extract_all_paths()
    
    # Extract just the path strings (the key is 'full_path', not 'path')
    taxonomy_paths = [item['full_path'] for item in taxonomy_data]
    print(f"Loaded {len(taxonomy_paths)} taxonomy paths")
    
    # Upload to Pinecone
    db.populate_from_taxonomy(taxonomy_paths)
    
    # Test search
    results = db.search("machine learning neural networks", top_k=5)
    print("\nTop 5 matches:")
    for path, score in results:
        print(f"  {score:.3f}: {path}")
    
    # Show stats
    print(f"\nDatabase stats: {db.get_stats()}")
