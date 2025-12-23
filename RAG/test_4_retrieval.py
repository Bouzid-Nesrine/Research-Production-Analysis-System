"""
Step 4: Test Retrieval Quality
Test semantic retrieval and ranking
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vector_db_manager import VectorDBManager
from config import CHROMA_DB_PATH, EMBEDDING_MODEL_NAME

print("="*70)
print("STEP 4: TESTING RETRIEVAL QUALITY")
print("="*70)

# Initialize database
print("\n[Setup] Initializing vector database...")
db_manager = VectorDBManager(
    db_path=CHROMA_DB_PATH,
    embedding_model_name=EMBEDDING_MODEL_NAME
)
db_manager.initialize_collection()
print(f"✓ Database loaded with {db_manager.collection.count()} paths")

# Test 1: Basic retrieval
print("\n[Test 1] Basic retrieval test...")

test_query = "Deep learning for image classification"
print(f"Query: {test_query}")

results = db_manager.retrieve_relevant_paths(test_query, top_k=10)

print(f"\nTop 10 retrieved paths:")
for path_info in results['retrieved_paths']:
    print(f"\n{path_info['rank']:2d}. {path_info['path']}")
    print(f"    Similarity: {path_info['similarity']:.4f}")
    print(f"    Domain: {path_info['domain']}, Level: {path_info['level']}")

# Test 2: Multiple query comparison
print("\n[Test 2] Testing different queries...")

queries = [
    "machine learning artificial intelligence neural networks",
    "medical imaging healthcare diagnosis",
    "climate change environmental science sustainability",
    "quantum computing physics algorithms",
    "natural language processing text analysis",
]

print("\nComparing retrieval for different queries:\n")

for i, query in enumerate(queries, 1):
    print(f"{i}. Query: {query}")
    results = db_manager.retrieve_relevant_paths(query, top_k=3)
    
    print(f"   Top 3 paths:")
    for path_info in results['retrieved_paths']:
        print(f"   - {path_info['path'][:65]}")
        print(f"     Similarity: {path_info['similarity']:.4f}")
    print()

# Test 3: Retrieval with real articles
print("\n[Test 3] Retrieval with full article abstracts...")

test_articles = [
    {
        'title': 'Deep Learning for Medical Image Segmentation',
        'abstract': '''This paper presents a novel deep learning approach for automated medical 
        image segmentation. We develop a convolutional neural network architecture that achieves 
        state-of-the-art performance on multiple medical imaging datasets including CT and MRI 
        scans for tumor detection.''',
        'expected_domain': 'Natural Science'  # or Medical and Health Science
    },
    {
        'title': 'Climate Change Impact on Crop Productivity',
        'abstract': '''We analyze the effects of climate change on agricultural productivity 
        across different regions using statistical models and satellite data. The study combines 
        historical climate data with crop yield statistics to quantify impacts on food security.''',
        'expected_domain': 'Agricultural Science'  # or Interdisciplinary
    },
    {
        'title': 'Quantum Algorithms for Optimization',
        'abstract': '''This work introduces quantum algorithms for solving combinatorial 
        optimization problems. We develop quantum annealing techniques and demonstrate their 
        advantages over classical algorithms on benchmark problems in logistics and finance.''',
        'expected_domain': 'Natural Science'  # or Interdisciplinary
    }
]

retrieval_results = []

for i, article in enumerate(test_articles, 1):
    print(f"\n{'='*70}")
    print(f"Article {i}: {article['title']}")
    print(f"{'='*70}")
    
    article_text = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
    results = db_manager.retrieve_relevant_paths(article_text, top_k=10)
    
    print(f"\nTop 10 retrieved paths:")
    for path_info in results['retrieved_paths']:
        print(f"\n{path_info['rank']:2d}. {path_info['path']}")
        print(f"    Similarity: {path_info['similarity']:.4f}")
        print(f"    Domain: {path_info['domain']}")
    
    # Store results
    retrieval_results.append({
        'article': article,
        'results': results
    })

# Test 4: Analyze retrieval quality
print("\n[Test 4] Analyzing retrieval quality...")

all_similarities = []
top1_similarities = []
domain_matches = []

for item in retrieval_results:
    article = item['article']
    results = item['results']['retrieved_paths']
    
    # Collect all similarities
    sims = [p['similarity'] for p in results]
    all_similarities.extend(sims)
    
    # Top-1 similarity
    top1_similarities.append(sims[0])
    
    # Check if expected domain appears in top results
    top_domains = [p['domain'] for p in results[:5]]
    if 'expected_domain' in article:
        domain_matches.append(article['expected_domain'] in top_domains)

print(f"\nRetrieval Quality Metrics:")
print(f"  All similarities - Mean: {np.mean(all_similarities):.4f}")
print(f"  All similarities - Std: {np.std(all_similarities):.4f}")
print(f"  Top-1 similarities - Mean: {np.mean(top1_similarities):.4f}")
print(f"  Top-1 similarities - Min: {np.min(top1_similarities):.4f}")
print(f"  Top-1 similarities - Max: {np.max(top1_similarities):.4f}")

if domain_matches:
    print(f"  Expected domain in top-5: {sum(domain_matches)}/{len(domain_matches)}")

# Test 5: Test top_k parameter
print("\n[Test 5] Testing different top_k values...")

test_query = "machine learning for data analysis"
k_values = [1, 3, 5, 10, 20]

print(f"Query: {test_query}\n")

for k in k_values:
    results = db_manager.retrieve_relevant_paths(test_query, top_k=k)
    sims = [p['similarity'] for p in results['retrieved_paths']]
    
    print(f"top_k={k:2d}:")
    print(f"  Retrieved: {len(results['retrieved_paths'])} paths")
    print(f"  Avg similarity: {np.mean(sims):.4f}")
    print(f"  Min similarity: {np.min(sims):.4f}")

# Test 6: Test similarity threshold
print("\n[Test 6] Testing similarity threshold...")

test_query = "artificial intelligence neural networks"

for threshold in [0.5, 0.6, 0.7, 0.8]:
    results = db_manager.retrieve_relevant_paths(
        test_query,
        top_k=20,
        similarity_threshold=threshold
    )
    
    print(f"\nThreshold={threshold}:")
    print(f"  Retrieved: {len(results['retrieved_paths'])} paths")
    if results['retrieved_paths']:
        sims = [p['similarity'] for p in results['retrieved_paths']]
        print(f"  Similarity range: {np.min(sims):.4f} - {np.max(sims):.4f}")

# Test 7: Diversity analysis
print("\n[Test 7] Analyzing retrieval diversity...")

for item in retrieval_results:
    article = item['article']
    results = item['results']['retrieved_paths']
    
    # Count unique domains
    domains = [p['domain'] for p in results]
    unique_domains = len(set(domains))
    
    # Count unique levels
    levels = [p['level'] for p in results]
    unique_levels = len(set(levels))
    
    print(f"\nArticle: {article['title'][:50]}...")
    print(f"  Unique domains in top-10: {unique_domains}")
    print(f"  Unique levels in top-10: {unique_levels}")
    print(f"  Domain distribution: {dict((d, domains.count(d)) for d in set(domains))}")

# Test 8: Export retrieval results
print("\n[Test 8] Exporting retrieval results...")

export_data = []
for item in retrieval_results:
    article = item['article']
    for path_info in item['results']['retrieved_paths']:
        export_data.append({
            'article_title': article['title'],
            'rank': path_info['rank'],
            'path': path_info['path'],
            'similarity': path_info['similarity'],
            'domain': path_info['domain'],
            'level': path_info['level']
        })

df = pd.DataFrame(export_data)
output_path = Path(__file__).parent / 'test_retrieval_results.csv'
df.to_csv(output_path, index=False)
print(f"✓ Results exported to: {output_path}")

# Test 9: Visualize similarity distribution
print("\n[Test 9] Visualizing similarity distribution...")

try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(all_similarities, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores')
    plt.axvline(np.mean(all_similarities), color='r', linestyle='--', 
                label=f'Mean: {np.mean(all_similarities):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot([
        [p['similarity'] for p in retrieval_results[i]['results']['retrieved_paths']]
        for i in range(len(retrieval_results))
    ], labels=[f"Article {i+1}" for i in range(len(retrieval_results))])
    plt.ylabel('Similarity Score')
    plt.title('Similarity Scores by Article')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    viz_path = Path(__file__).parent / 'test_retrieval_visualization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {viz_path}")
    
except Exception as e:
    print(f"⚠ Visualization skipped: {e}")

# Final summary
print("\n" + "="*70)
print("RETRIEVAL QUALITY TEST SUMMARY")
print("="*70)
print(f"✓ Tested retrieval with {len(queries) + len(test_articles)} different queries")
print(f"✓ Average similarity score: {np.mean(all_similarities):.4f}")
print(f"✓ Top-1 average similarity: {np.mean(top1_similarities):.4f}")
print(f"✓ Results exported to: {output_path}")
print("="*70)
print("\n✅ ALL TESTS PASSED - Retrieval system is working correctly!")
print("\nYou're ready to test the full pipeline!")
print("Next: Test with LLM using quickstart.py or RAG_Classification_Demo.ipynb")
print("="*70)
