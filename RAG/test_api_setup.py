"""
Test Alibaba Cloud API Setup
Tests API connection and basic functionality
"""

import os
import sys
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_env_file():
    """Test 1: Check if .env file exists and loads"""
    print("\n" + "="*60)
    print("TEST 1: Environment File")
    print("="*60)
    
    # Load .env
    load_dotenv()
    
    # Check if API key exists
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("✗ FAILED: GOOGLE_API_KEY not found in environment")
        print("\nPlease create a .env file with your API key:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your API key: GOOGLE_API_KEY=your-key-here")
        print("  3. Get API key from: https://aistudio.google.com/app/apikey")
        return False
    
    # Mask key for security
    masked_key = api_key[:8] + "..." + api_key[-4:]
    print(f"✓ PASSED: API key found: {masked_key}")
    return True

def test_llm_import():
    """Test 2: Import LLM classifier"""
    print("\n" + "="*60)
    print("TEST 2: Import LLM Classifier")
    print("="*60)
    
    try:
        from RAG.llm_classifier_api import LLMClassifier
        print("✓ PASSED: LLMClassifier imported successfully")
        return True
    except ImportError as e:
        print(f"✗ FAILED: Cannot import LLMClassifier: {e}")
        print("\nMake sure you installed requirements:")
        print("  pip install -r requirements.txt")
        return False

def test_llm_initialization():
    """Test 3: Initialize LLM classifier"""
    print("\n" + "="*60)
    print("TEST 3: Initialize LLM Classifier")
    print("="*60)
    
    try:
        from RAG.llm_classifier_api import LLMClassifier
        
        # Use Google Gemini model
        classifier = LLMClassifier(model_name="gemini-2.5-flash-lite")
        print("✓ PASSED: LLMClassifier initialized successfully")
        print(f"  Model: {classifier.model_name}")
        print(f"  SDK: Google Generative AI")
        return True, classifier
    except ValueError as e:
        print(f"✗ FAILED: {e}")
        return False, None
    except Exception as e:
        print(f"✗ FAILED: Unexpected error: {e}")
        return False, None

def test_api_call(classifier):
    """Test 4: Make a simple API call"""
    print("\n" + "="*60)
    print("TEST 4: API Call")
    print("="*60)
    
    try:
        # Simple test prompt
        prompt = "What is artificial intelligence? Respond in one sentence."
        
        print("Sending test request to API...")
        response = classifier.classify(
            prompt=prompt,
            temperature=0.5,
            max_tokens=50
        )
        
        print(f"✓ PASSED: API call successful")
        print(f"  Response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"✗ FAILED: API call failed: {e}")
        
        if "AccessDenied.Unpurchased" in str(e):
            print("\n⚠️  Model Access Issue:")
            print(f"  The model '{classifier.model_name}' is not enabled in your account.")
            print("\nHow to fix:")
            print("  1. Visit: https://modelstudio.console.alibabacloud.com/")
            print("  2. Enable/activate the model (may be free or require purchase)")
            print("  3. Or change model in config.py to one you have access to")
        else:
            print("\nPossible issues:")
            print("  1. Check your internet connection")
            print("  2. Verify API key is correct and active")
            print("  3. Ensure DashScope API is enabled in your account")
        return False

def test_classification(classifier):
    """Test 5: Test article classification"""
    print("\n" + "="*60)
    print("TEST 5: Article Classification")
    print("="*60)
    
    try:
        # Sample article
        title = "Machine Learning for Medical Diagnosis"
        abstract = "This paper presents a machine learning approach for automated medical diagnosis using patient data."
        
        # Sample taxonomy paths
        relevant_paths = [
            "Natural Science > Computer and Information Science > Artificial Intelligence > Machine Learning",
            "Medical and Health Science > Clinical Medicine > Internal Medicine",
            "Natural Science > Computer and Information Science > Artificial Intelligence > Computer Vision",
        ]
        
        print("Classifying test article...")
        result = classifier.classify_article(
            title=title,
            abstract=abstract,
            relevant_paths=relevant_paths
        )
        
        print(f"✓ PASSED: Classification successful")
        print(f"  Path: {result['classification']['path']}")
        print(f"  Confidence: {result['classification']['confidence']}")
        if result['classification']['reasoning']:
            print(f"  Reasoning: {result['classification']['reasoning'][:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: Classification failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ALIBABA CLOUD API SETUP TEST")
    print("="*60)
    
    # Test 1: Environment file
    if not test_env_file():
        print("\n⚠️  Please fix environment setup before continuing")
        sys.exit(1)
    
    # Test 2: Import
    if not test_llm_import():
        print("\n⚠️  Please install dependencies before continuing")
        sys.exit(1)
    
    # Test 3: Initialize
    success, classifier = test_llm_initialization()
    if not success:
        print("\n⚠️  API initialization failed")
        sys.exit(1)
    
    # Test 4: Simple API call
    if not test_api_call(classifier):
        print("\n⚠️  API call failed - check your API key and connection")
        sys.exit(1)
    
    # Test 5: Classification
    if not test_classification(classifier):
        print("\n⚠️  Classification test failed")
        sys.exit(1)
    
    # All tests passed
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour Alibaba Cloud API is configured correctly!")
    print("You can now use the RAG classification pipeline.")
    print("\nNext steps:")
    print("  1. Run setup_pipeline.py to initialize the database")
    print("  2. Run quickstart.py to classify example articles")
    print("  3. Use rag_pipeline.py for your own articles")
    
if __name__ == "__main__":
    main()
