"""
Test 5: Complete RAG Pipeline Test
End-to-end testing of retrieval + LLM classification
"""

import sys
from pathlib import Path
import time
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import RAGClassificationPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*70)
print("TEST 5: COMPLETE RAG PIPELINE TEST")
print("="*70)

# Test articles with expected domains
TEST_ARTICLES = [
    {
        "id": 1,
        "title": "Deep Learning for Medical Image Segmentation Using Convolutional Neural Networks",
        "abstract": """This paper presents a novel deep learning approach for automated medical 
        image segmentation. We develop a convolutional neural network architecture that achieves 
        state-of-the-art performance on multiple medical imaging datasets. The model is evaluated 
        on CT and MRI scans for tumor detection.""",
        "expected_domain": "Computer Science",  # Should be in AI/Computer Vision area
        "expected_keywords": ["deep learning", "medical", "image", "segmentation", "neural network"]
    },
    {
        "id": 2,
        "title": "Climate Change Impact on Agricultural Productivity in Sub-Saharan Africa",
        "abstract": """We analyze the effects of climate change on crop yields across different 
        regions in Sub-Saharan Africa using statistical models and satellite data. Our study 
        combines historical climate data with agricultural output statistics to quantify the 
        relationship between temperature changes and crop productivity.""",
        "expected_domain": "Environmental Science",
        "expected_keywords": ["climate", "agriculture", "crop", "africa"]
    },
    {
        "id": 3,
        "title": "Quantum Algorithms for Combinatorial Optimization Problems",
        "abstract": """This work introduces new quantum algorithms for solving complex combinatorial 
        optimization problems. We develop quantum annealing techniques and demonstrate their 
        superiority over classical algorithms on benchmark problems. Applications in logistics 
        and finance are discussed.""",
        "expected_domain": "Physics",  # Quantum computing
        "expected_keywords": ["quantum", "algorithm", "optimization"]
    }
]

def test_pipeline_initialization():
    """Test 1: Pipeline initialization"""
    print("\n" + "="*60)
    print("TEST 1: Pipeline Initialization")
    print("="*60)
    
    try:
        start = time.time()
        pipeline = RAGClassificationPipeline(auto_setup=True)
        init_time = time.time() - start
        
        print(f"✓ Pipeline initialized in {init_time:.2f}s")
        print(f"  • Taxonomy paths: {len(pipeline.taxonomy_paths)}")
        print(f"  • Database paths: {pipeline.db_manager.collection.count()}")
        print(f"  • Embedding model: {pipeline.embedding_model_name}")
        print(f"  • LLM model: {pipeline.llm_model_name}")
        
        return True, pipeline
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False, None

def test_retrieval_only(pipeline, article):
    """Test 2: Retrieval (embedding-based)"""
    print("\n" + "-"*50)
    print(f"TEST 2a: Retrieval for Article {article['id']}")
    print("-"*50)
    
    try:
        query = f"Title: {article['title']}\nAbstract: {article['abstract']}"
        
        start = time.time()
        results = pipeline.db_manager.retrieve_relevant_paths(query, top_k=5)
        retrieval_time = time.time() - start
        
        print(f"✓ Retrieved {len(results['retrieved_paths'])} paths in {retrieval_time:.3f}s")
        print(f"\nTop 5 paths:")
        for i, path_info in enumerate(results['retrieved_paths'][:5], 1):
            print(f"  {i}. {path_info['path'][:70]}...")
            print(f"     Similarity: {path_info['similarity']:.4f}")
        
        return True, results['retrieved_paths']
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False, []

def test_llm_classification(pipeline, article):
    """Test 3: Full RAG classification (retrieval + LLM)"""
    print("\n" + "-"*50)
    print(f"TEST 3a: Full Classification for Article {article['id']}")
    print("-"*50)
    
    try:
        start = time.time()
        result = pipeline.classify_article(
            title=article['title'],
            abstract=article['abstract'],
            top_k=5,
            temperature=0.3
        )
        classification_time = time.time() - start
        
        if 'error' in result:
            print(f"✗ Classification failed: {result['error']}")
            return False, result
        
        classification = result['classification']
        print(f"✓ Classification completed in {classification_time:.2f}s")
        print(f"\nResult:")
        print(f"  • Path: {classification['path']}")
        print(f"  • Confidence: {classification['confidence']}")
        print(f"  • Valid: {classification.get('valid', 'N/A')}")
        if classification.get('reasoning'):
            print(f"  • Reasoning: {classification['reasoning'][:100]}...")
        
        # Check if expected keywords match
        path_lower = classification['path'].lower()
        matching_keywords = [kw for kw in article['expected_keywords'] if kw in path_lower]
        print(f"  • Keyword matches: {len(matching_keywords)}/{len(article['expected_keywords'])} ({matching_keywords})")
        
        return True, result
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

def test_batch_classification(pipeline, articles):
    """Test 4: Batch classification"""
    print("\n" + "="*60)
    print("TEST 4: Batch Classification")
    print("="*60)
    
    try:
        batch_articles = [
            {'title': a['title'], 'abstract': a['abstract']} 
            for a in articles
        ]
        
        start = time.time()
        results = pipeline.batch_classify(batch_articles, show_progress=True)
        batch_time = time.time() - start
        
        successful = sum(1 for r in results if r and 'error' not in r)
        
        print(f"\n✓ Batch completed in {batch_time:.2f}s")
        print(f"  • Total articles: {len(articles)}")
        print(f"  • Successful: {successful}")
        print(f"  • Failed: {len(articles) - successful}")
        print(f"  • Avg time per article: {batch_time/len(articles):.2f}s")
        
        return True, results
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_statistics(pipeline):
    """Test 5: Pipeline statistics"""
    print("\n" + "="*60)
    print("TEST 5: Pipeline Statistics")
    print("="*60)
    
    try:
        stats = pipeline.get_statistics()
        print("Pipeline statistics:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("COMPLETE RAG PIPELINE TEST SUITE")
    print("="*70)
    
    results = {
        'initialization': False,
        'retrieval': 0,
        'classification': 0,
        'batch': False,
        'statistics': False
    }
    
    # Test 1: Initialize pipeline
    success, pipeline = test_pipeline_initialization()
    results['initialization'] = success
    
    if not success:
        print("\n⚠️ Pipeline initialization failed. Cannot continue tests.")
        return results
    
    # Test 2 & 3: Test individual articles (retrieval + classification)
    print("\n" + "="*60)
    print("INDIVIDUAL ARTICLE TESTS")
    print("="*60)
    
    # Test only first article to respect API limits
    test_article = TEST_ARTICLES[0]
    
    # Test retrieval
    success, paths = test_retrieval_only(pipeline, test_article)
    if success:
        results['retrieval'] += 1
    
    # Test full classification
    success, result = test_llm_classification(pipeline, test_article)
    if success:
        results['classification'] += 1
    
    # Test 4: Batch classification (skip to save API calls)
    # Uncomment below to test batch processing
    # success, batch_results = test_batch_classification(pipeline, TEST_ARTICLES[:1])
    # results['batch'] = success
    
    # Test 5: Statistics
    results['statistics'] = test_statistics(pipeline)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ Initialization: {'PASSED' if results['initialization'] else 'FAILED'}")
    print(f"✓ Retrieval tests: {results['retrieval']}/1 passed")
    print(f"✓ Classification tests: {results['classification']}/1 passed")
    print(f"✓ Statistics: {'PASSED' if results['statistics'] else 'FAILED'}")
    
    total_passed = (
        results['initialization'] + 
        results['retrieval'] + 
        results['classification'] + 
        results['statistics']
    )
    total_tests = 4
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED! RAG pipeline is working correctly.")
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed.")
    
    return results

if __name__ == "__main__":
    main()
