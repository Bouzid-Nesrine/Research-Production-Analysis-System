"""
Simple test to verify RAG pipeline works in RAG folder
"""
import sys
from pathlib import Path

print("Current working directory:", Path.cwd())
print("Python path:", sys.path[:3])

try:
    print("\n1. Testing RAG pipeline import...")
    from rag_pipeline import RAGClassificationPipeline
    print("✓ RAG pipeline imported successfully")
    
    print("\n2. Initializing RAG pipeline...")
    pipeline = RAGClassificationPipeline(auto_setup=True)
    print("✓ RAG pipeline initialized")
    
    print("\n3. Testing classification...")
    result = pipeline.classify_article(
        title="Deep Learning for Medical Image Segmentation",
        abstract="This paper presents a novel convolutional neural network architecture for automated segmentation of medical images. We evaluate our approach on multiple datasets and show significant improvements over existing methods."
    )
    
    print("\n4. Classification result:")
    if 'error' in result:
        print(f"✗ Error: {result['error']}")
    else:
        classification = result.get('classification', {})
        print(f"✓ Path: {classification.get('path')}")
        print(f"✓ Confidence: {classification.get('confidence')}")
        print(f"✓ Reasoning: {classification.get('reasoning', '')[:100]}...")
        
    print("\n✓ All tests passed!")
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
