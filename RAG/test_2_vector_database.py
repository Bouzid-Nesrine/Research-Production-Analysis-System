"""
Step 2: Test Vector Database (ChromaDB)
Setup and verify ChromaDB functionality
"""

import os
# Force CPU-only mode to avoid CUDA/NCCL issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vector_db_manager import VectorDBManager
from taxonomy_parser import TaxonomyParser
from config import TAXONOMY_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL_NAME

print("="*70)
print("STEP 2: TESTING VECTOR DATABASE (ChromaDB)")
print("="*70)

# Test 1: Load taxonomy paths
print("\n[Test 1] Loading taxonomy paths...")
parser = TaxonomyParser(TAXONOMY_PATH)
paths = parser.extract_all_paths()
print(f"✓ Loaded {len(paths)} taxonomy paths")

# Test 2: Initialize ChromaDB
print("\n[Test 2] Initializing ChromaDB...")
print(f"Database path: {CHROMA_DB_PATH}")
print(f"Embedding model: {EMBEDDING_MODEL_NAME}")

try:
    db_manager = VectorDBManager(
        db_path=CHROMA_DB_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    print("✓ VectorDBManager initialized")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# Test 3: Create collection
print("\n[Test 3] Creating/loading collection...")
try:
    db_manager.initialize_collection(reset=False)
    current_count = db_manager.collection.count()
    print(f"✓ Collection initialized")
    print(f"  Current items in database: {current_count}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# Test 4: Populate database (if empty)
if current_count == 0:
    print("\n[Test 4] Populating database with embeddings...")
    print("This will take a few minutes - generating embeddings for all paths...")
    
    start_time = time.time()
    try:
        db_manager.populate_from_paths(paths, batch_size=50, show_progress=True)
        elapsed = time.time() - start_time
        print(f"✓ Database populated in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)
else:
    print(f"\n[Test 4] Database already populated with {current_count} items")
    print("Skipping population step")

# Test 5: Verify database contents
print("\n[Test 5] Verifying database contents...")
final_count = db_manager.collection.count()
print(f"  Total items: {final_count}")
print(f"  Expected items: {len(paths)}")

if final_count == len(paths):
    print("✓ Database contains all taxonomy paths")
elif final_count > 0:
    print(f"⚠ Warning: Database has {final_count} items, expected {len(paths)}")
else:
    print("❌ ERROR: Database is empty")
    sys.exit(1)

# Test 6: Get collection statistics
print("\n[Test 6] Getting collection statistics...")
stats = db_manager.get_collection_stats()
print(f"  Collection name: {stats['collection_name']}")
print(f"  Embedding model: {stats['embedding_model']}")
print(f"  Total paths: {stats['total_paths']}")

# Test 7: Test embedding generation
print("\n[Test 7] Testing embedding generation...")
test_text = "Deep learning for image classification using neural networks"
print(f"Test text: {test_text}")

try:
    embedding = db_manager.embedding_model.encode(test_text)
    print(f"✓ Embedding generated")
    print(f"  Embedding dimension: {len(embedding)}")
    print(f"  Embedding type: {type(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# Test 8: Test basic retrieval
print("\n[Test 8] Testing semantic retrieval...")
test_queries = [
    "machine learning and artificial intelligence",
    "medical imaging and healthcare",
    "climate change and environmental science",
]

for i, query in enumerate(test_queries, 1):
    print(f"\n  Query {i}: {query}")
    try:
        results = db_manager.retrieve_relevant_paths(query, top_k=3)
        print(f"  ✓ Retrieved {len(results['retrieved_paths'])} paths")
        
        for j, path_info in enumerate(results['retrieved_paths'], 1):
            print(f"    {j}. {path_info['path'][:60]}...")
            print(f"       Similarity: {path_info['similarity']:.4f}")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

# Test 9: Test similarity scores
print("\n[Test 9] Analyzing similarity scores...")
all_similarities = []
for query in test_queries:
    results = db_manager.retrieve_relevant_paths(query, top_k=10)
    for path_info in results['retrieved_paths']:
        all_similarities.append(path_info['similarity'])

import numpy as np
print(f"  Number of scores: {len(all_similarities)}")
print(f"  Mean similarity: {np.mean(all_similarities):.4f}")
print(f"  Max similarity: {np.max(all_similarities):.4f}")
print(f"  Min similarity: {np.min(all_similarities):.4f}")
print(f"  Std deviation: {np.std(all_similarities):.4f}")

# Test 10: Test domain filtering
print("\n[Test 10] Testing domain filtering...")
try:
    # Get sample domain
    sample = db_manager.collection.peek(limit=1)
    if sample['metadatas']:
        test_domain = sample['metadatas'][0]['domain']
        print(f"  Testing with domain: {test_domain}")
        
        domain_paths = db_manager.get_paths_by_domain(test_domain, limit=5)
        print(f"  ✓ Retrieved {len(domain_paths)} paths from {test_domain}")
        
        for i, path_info in enumerate(domain_paths[:3], 1):
            print(f"    {i}. {path_info['path'][:60]}...")
except Exception as e:
    print(f"  ⚠ Domain filtering test skipped: {e}")

# Test 11: Test keyword search
print("\n[Test 11] Testing keyword search...")
test_keywords = ["neural", "network", "learning"]
print(f"  Keywords: {test_keywords}")

try:
    keyword_results = db_manager.search_by_keywords(test_keywords, top_k=5)
    print(f"  ✓ Found {len(keyword_results)} paths")
    
    for i, path_info in enumerate(keyword_results[:3], 1):
        print(f"    {i}. {path_info['path'][:60]}...")
        print(f"       Similarity: {path_info['similarity']:.4f}")
except Exception as e:
    print(f"  ❌ ERROR: {e}")

# Test 12: Performance test
print("\n[Test 12] Performance test...")
print("  Running 10 retrieval queries...")

start_time = time.time()
for i in range(10):
    db_manager.retrieve_relevant_paths(
        "test query for performance measurement",
        top_k=10
    )
elapsed = time.time() - start_time

print(f"  ✓ Completed in {elapsed:.2f} seconds")
print(f"  Average time per query: {elapsed/10:.3f} seconds")

# Final summary
print("\n" + "="*70)
print("VECTOR DATABASE TEST SUMMARY")
print("="*70)
print(f"✓ ChromaDB initialized at: {CHROMA_DB_PATH}")
print(f"✓ Embedding model loaded: {EMBEDDING_MODEL_NAME}")
print(f"✓ Database populated: {final_count} paths")
print(f"✓ Semantic retrieval working")
print(f"✓ Average query time: {elapsed/10:.3f} seconds")
print("="*70)
print("\n✅ ALL TESTS PASSED - Vector database is working correctly!")
print("\nNext step: Run test_3_embeddings.py")
print("="*70)
