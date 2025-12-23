"""
Quick Start Guide for RAG Classification Pipeline
"""

# =============================================================================
# STEP 1: Setup Pipeline
# =============================================================================

print("=" * 60)
print("RAG CLASSIFICATION PIPELINE - QUICK START")
print("=" * 60)

# First, run setup (one-time only)
# python setup_pipeline.py

# =============================================================================
# STEP 2: Import and Initialize
# =============================================================================

from rag_pipeline import RAGClassificationPipeline

# Initialize pipeline (will load existing database)
print("\nInitializing pipeline...")
pipeline = RAGClassificationPipeline(auto_setup=True)

print("✓ Pipeline ready!")

# =============================================================================
# STEP 3: Example Articles
# =============================================================================

example_articles = [
    {
        'id': 'article_1',
        'title': 'Deep Learning for Medical Image Segmentation Using Convolutional Neural Networks',
        'abstract': '''This paper presents a novel deep learning approach for automated medical 
        image segmentation. We develop a convolutional neural network architecture that achieves 
        state-of-the-art performance on multiple medical imaging datasets. The model is evaluated 
        on CT and MRI scans for tumor detection and achieves significant improvements over existing 
        methods. Our results demonstrate the potential of deep learning in clinical applications.'''
    },
    {
        'id': 'article_2',
        'title': 'Climate Change Impact on Agricultural Productivity in Sub-Saharan Africa',
        'abstract': '''We analyze the effects of climate change on crop yields across different 
        regions in Sub-Saharan Africa using statistical models and satellite data. Our study 
        combines historical climate data with agricultural output statistics to quantify the 
        relationship between temperature changes, precipitation patterns, and crop productivity. 
        The results show significant negative impacts on major staple crops.'''
    },
    {
        'id': 'article_3',
        'title': 'Quantum Algorithms for Combinatorial Optimization Problems',
        'abstract': '''This work introduces new quantum algorithms for solving complex combinatorial 
        optimization problems with applications in logistics and finance. We develop quantum 
        annealing techniques and demonstrate their superiority over classical algorithms on 
        benchmark problems. The algorithms are tested on quantum simulators and show promising 
        results for practical quantum computing applications.'''
    },
    {
        'id': 'article_4',
        'title': 'Social Media Sentiment Analysis for Political Opinion Mining',
        'abstract': '''This study presents a natural language processing framework for analyzing 
        political sentiment on social media platforms. We collect and analyze millions of tweets 
        during election campaigns, applying machine learning techniques for sentiment classification 
        and opinion trend detection. Our methods achieve high accuracy in predicting public opinion 
        shifts and political outcomes.'''
    },
    {
        'id': 'article_5',
        'title': 'Biodegradable Polymers for Sustainable Packaging Applications',
        'abstract': '''We investigate the synthesis and characterization of biodegradable polymers 
        derived from renewable resources for use in sustainable packaging. Various polymer 
        compositions are tested for mechanical strength, barrier properties, and biodegradation 
        rates. The results demonstrate that these bio-based materials can effectively replace 
        conventional petroleum-based plastics while reducing environmental impact.'''
    }
]

# =============================================================================
# STEP 4: Classify Single Article
# =============================================================================

print("\n" + "=" * 60)
print("EXAMPLE 1: Single Article Classification")
print("=" * 60)

article = example_articles[0]
print(f"\nArticle: {article['title'][:60]}...")

result = pipeline.classify_article(
    title=article['title'],
    abstract=article['abstract'],
    top_k=10
)

if 'error' not in result:
    print(f"\n✓ Classification Result:")
    print(f"  Path: {result['classification']['path']}")
    print(f"  Confidence: {result['classification']['confidence']}")
    print(f"  Reasoning: {result['classification']['reasoning']}")
    print(f"\n  Processing time: {result['metadata']['total_time']:.2f}s")
    print(f"  Retrieved paths: {result['metadata']['retrieval']['paths_retrieved']}")
    
    print(f"\n  Top 3 Retrieved Paths:")
    for i, (path, score) in enumerate(zip(
        result['metadata']['retrieval']['paths_retrieved'][:3],
        result['metadata']['retrieval']['similarity_scores'][:3]
    ), 1):
        # Path is in the retrieved_paths list from db_manager
        print(f"    {i}. Similarity: {score:.4f}")
else:
    print(f"\n❌ Error: {result['error']}")

# =============================================================================
# STEP 5: Batch Classification
# =============================================================================

print("\n" + "=" * 60)
print("EXAMPLE 2: Batch Classification")
print("=" * 60)

print(f"\nClassifying {len(example_articles)} articles...")

results = pipeline.batch_classify(
    example_articles,
    show_progress=True
)

# Display results
print(f"\n{'='*60}")
print("CLASSIFICATION RESULTS")
print(f"{'='*60}\n")

for i, (article, result) in enumerate(zip(example_articles, results), 1):
    print(f"{i}. {article['title'][:50]}...")
    
    if 'error' not in result:
        print(f"   ✓ Path: {result['classification']['path']}")
        print(f"   Confidence: {result['classification']['confidence']}")
        print(f"   Time: {result['metadata']['total_time']:.2f}s")
    else:
        print(f"   ❌ Error: {result['error']}")
    print()

# =============================================================================
# STEP 6: Save Results
# =============================================================================

print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

from pathlib import Path

output_dir = Path(__file__).parent / 'results'
output_dir.mkdir(exist_ok=True)

# Save as JSON
json_path = output_dir / 'classification_results.json'
pipeline.save_results(results, json_path, format='json')
print(f"✓ Saved JSON: {json_path}")

# Save as CSV
csv_path = output_dir / 'classification_results.csv'
pipeline.save_results(results, csv_path, format='csv')
print(f"✓ Saved CSV: {csv_path}")

# =============================================================================
# STEP 7: Pipeline Statistics
# =============================================================================

print("\n" + "=" * 60)
print("PIPELINE STATISTICS")
print("=" * 60)

stats = pipeline.get_statistics()
print(f"\nTotal classified: {stats['total_classified']}")
print(f"Successful: {stats['successful']}")
print(f"Failed: {stats['failed']}")
print(f"Success rate: {stats['success_rate']:.1%}")

# =============================================================================
# STEP 8: Test Different Parameters
# =============================================================================

print("\n" + "=" * 60)
print("EXAMPLE 3: Hyperparameter Testing")
print("=" * 60)

test_article = example_articles[0]

print("\nTesting different top_k values...")
for k in [5, 10, 15]:
    result = pipeline.classify_article(
        title=test_article['title'],
        abstract=test_article['abstract'],
        top_k=k,
        return_metadata=True
    )
    
    if 'error' not in result:
        print(f"\ntop_k={k}:")
        print(f"  Path: {result['classification']['path']}")
        print(f"  Time: {result['metadata']['total_time']:.2f}s")

print("\nTesting different temperatures...")
for temp in [0.1, 0.3, 0.7]:
    result = pipeline.classify_article(
        title=test_article['title'],
        abstract=test_article['abstract'],
        temperature=temp,
        return_metadata=True
    )
    
    if 'error' not in result:
        print(f"\ntemperature={temp}:")
        print(f"  Path: {result['classification']['path']}")
        print(f"  Confidence: {result['classification']['confidence']}")

# =============================================================================
# STEP 9: Retrieval Testing
# =============================================================================

print("\n" + "=" * 60)
print("EXAMPLE 4: Retrieval Testing")
print("=" * 60)

query = f"Title: {test_article['title']}\n\nAbstract: {test_article['abstract']}"

retrieval_results = db_manager.retrieve_relevant_paths(
    query_text=query,
    top_k=10
)

print(f"\nTop 10 retrieved paths for: '{test_article['title'][:50]}...'\n")
for path_info in retrieval_results['retrieved_paths']:
    print(f"{path_info['rank']}. {path_info['path']}")
    print(f"   Similarity: {path_info['similarity']:.4f}")
    print()

print("\n" + "=" * 60)
print("QUICK START COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("1. Process your own articles")
print("2. Tune hyperparameters (top_k, temperature)")
print("3. Evaluate on test set")
print("4. See PIPELINE.md for detailed documentation")
print("=" * 60)
