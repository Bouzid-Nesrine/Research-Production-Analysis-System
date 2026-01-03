#!/usr/bin/env python3
"""
Quick test script for local SciBERT model
"""

import sys
from pathlib import Path

# Add RAG folder to path
sys.path.insert(0, str(Path(__file__).parent))

from local_model_classifier import LocalModelClassifier

def test_local_model():
    """Test the local model classifier"""
    
    print("="*80)
    print("Testing Local SciBERT Model with LoRA")
    print("="*80)
    
    # Initialize model
    print("\n1. Loading model...")
    classifier = LocalModelClassifier()
    
    print(f"✓ Model loaded: {classifier.model_name}")
    print(f"✓ Device: {classifier.device}")
    print(f"✓ Number of labels: {classifier.num_labels}")
    
    # Test article
    test_article = {
        'title': 'Deep Learning for Medical Image Segmentation',
        'abstract': 'This paper presents a comprehensive review of deep learning techniques '
                   'for medical image segmentation. We analyze convolutional neural networks (CNNs), '
                   'U-Net architectures, and transformer-based models for segmenting anatomical '
                   'structures in CT, MRI, and X-ray images.'
    }
    
    # Candidate paths from retrieval with similarity scores
    candidate_paths = [
        {'path': 'Natural Science > Computer and Information Science > Artificial Intelligence > Machine Learning > Deep Learning > Neural Network Architecture > Convolutional Neural Network', 'similarity': 0.48},
        {'path': 'Natural Science > Computer and Information Science > Artificial Intelligence > Computer Vision > Image Understanding > Semantic Segmentation', 'similarity': 0.38},
        {'path': 'Engineering and Technology > Medical Engineering > Biomedical Engineering > Medical Imaging > Computed Tomography Imaging', 'similarity': 0.37}
    ]
    
    print(f"\n2. Classifying test article...")
    print(f"   Title: {test_article['title']}")
    print(f"   Candidate paths: {len(candidate_paths)}")
    print(f"\n   Retrieved candidates with similarity scores:")
    for i, item in enumerate(candidate_paths, 1):
        print(f"     {i}. {item['path'][:80]}... (sim: {item['similarity']:.2f})")
    
    # Classify
    result = classifier.classify_article(
        title=test_article['title'],
        abstract=test_article['abstract'],
        relevant_paths=candidate_paths
    )
    
    print(f"\n3. Re-Ranking Results:")
    print(f"   ✓ Selected Path: {result['classification']['path'][:80]}...")
    print(f"   ✓ Confidence: {result['classification']['confidence']}")
    print(f"   ✓ Combined Score: {result['classification'].get('confidence_score', 0):.4f}")
    print(f"   ✓ Model Score: {result['classification'].get('model_score', 0):.4f}")
    print(f"   ✓ Retrieval Score: {result['classification'].get('retrieval_score', 0):.4f}")
    print(f"   ✓ Reasoning: {result['classification']['reasoning']}")
    
    print("\n" + "="*80)
    print("✅ Local model test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    test_local_model()
