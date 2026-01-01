"""
Test script for API key rotation functionality
"""

import os
from dotenv import load_dotenv
from llm_classifier import LLMClassifier
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_api_key_loading():
    """Test that API keys are loaded correctly"""
    load_dotenv()
    
    print("\n" + "="*60)
    print("API KEY CONFIGURATION TEST")
    print("="*60)
    
    # Check environment variables
    api_keys_str = os.getenv('GOOGLE_API_KEYS')
    single_key = os.getenv('GOOGLE_API_KEY')
    
    print(f"\nGOOGLE_API_KEYS: {api_keys_str[:50]}..." if api_keys_str else "Not set")
    print(f"GOOGLE_API_KEY: {single_key[:30]}..." if single_key else "Not set")
    
    # Initialize classifier
    try:
        classifier = LLMClassifier()
        print(f"\n✓ Classifier initialized successfully")
        print(f"  Total API keys loaded: {len(classifier.api_keys)}")
        print(f"  Current key index: {classifier.current_key_index + 1}")
        
        # Show key status
        status = classifier.get_api_key_status()
        print(f"\nAPI Key Status:")
        print(f"  Current key: {status['current_key_prefix']}")
        print(f"  Healthy keys: {status['healthy_keys']}/{status['total_keys']}")
        
    except Exception as e:
        print(f"\n✗ Failed to initialize classifier: {e}")
        return False
    
    return True


def test_simple_classification():
    """Test a simple classification with key rotation"""
    print("\n" + "="*60)
    print("SIMPLE CLASSIFICATION TEST")
    print("="*60)
    
    try:
        classifier = LLMClassifier()
        
        # Create a simple prompt
        prompt = """Classify this article:
Title: Deep Learning for Image Classification
Abstract: This paper presents a convolutional neural network...

Paths:
1. Natural Science > Computer Science > AI > Machine Learning
2. Engineering > Electrical Engineering > Signal Processing
3. Natural Science > Mathematics > Statistics

Reply: Path: [exact path] | Confidence: [High/Medium/Low]"""
        
        print("\nSending classification request...")
        print(f"Using API key #{classifier.current_key_index + 1}")
        
        response = classifier.classify(prompt, temperature=0.1, max_tokens=100)
        
        print(f"\n✓ Response received:")
        print(f"  {response}")
        
        # Show updated status
        status = classifier.get_api_key_status()
        print(f"\nPost-request status:")
        print(f"  Current key: #{status['current_key_index']}")
        print(f"  Failures: {status['failure_counts']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Classification failed: {e}")
        return False


def test_key_rotation_simulation():
    """Simulate key rotation behavior"""
    print("\n" + "="*60)
    print("KEY ROTATION SIMULATION")
    print("="*60)
    
    try:
        # Create classifier with test keys
        test_keys = ["key1", "key2", "key3", "key4"]
        classifier = LLMClassifier(api_keys=test_keys)
        
        print(f"\nInitialized with {len(classifier.api_keys)} test keys")
        print(f"Current key index: {classifier.current_key_index}")
        
        # Simulate rotation
        print("\nSimulating key rotations:")
        for i in range(5):
            success = classifier._rotate_api_key()
            if success:
                print(f"  Rotation {i+1}: Now using key #{classifier.current_key_index + 1}")
            else:
                print(f"  Rotation {i+1}: Failed (no keys available)")
        
        # Simulate failures
        print("\nSimulating failures on key #2:")
        classifier.current_key_index = 1
        classifier.key_failure_count[1] = 3
        
        status = classifier.get_api_key_status()
        print(f"  Failure count: {status['failure_counts']}")
        
        # Test rotation after failures
        classifier._rotate_api_key()
        print(f"  After rotation: Now on key #{classifier.current_key_index + 1}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("API KEY ROTATION TEST SUITE")
    print("="*60)
    
    results = {
        "Key Loading": test_api_key_loading(),
        "Simple Classification": test_simple_classification(),
        "Rotation Simulation": test_key_rotation_simulation()
    }
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
