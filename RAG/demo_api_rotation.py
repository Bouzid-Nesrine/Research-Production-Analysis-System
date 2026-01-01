"""
Quick demo of API key rotation functionality
Run this to see automatic key switching in action
"""

from llm_classifier import LLMClassifier
import logging

# Setup logging to see rotation messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def main():
    print("\n" + "="*70)
    print("API KEY ROTATION DEMO")
    print("="*70)
    
    # Initialize classifier
    print("\n1️⃣  Initializing classifier with multiple API keys...")
    classifier = LLMClassifier()
    
    status = classifier.get_api_key_status()
    print(f"   ✓ Loaded {status['total_keys']} API key(s)")
    print(f"   ✓ Currently using key #{status['current_key_index']}")
    print(f"   ✓ All keys healthy: {status['healthy_keys']}/{status['total_keys']}")
    
    # Show key rotation
    if len(classifier.api_keys) > 1:
        print("\n2️⃣  Testing key rotation...")
        print(f"   Starting with key #{classifier.current_key_index + 1}")
        
        for i in range(min(3, len(classifier.api_keys))):
            classifier._rotate_api_key()
            print(f"   → Rotated to key #{classifier.current_key_index + 1}")
    else:
        print("\n2️⃣  Only one key available (add more to see rotation)")
    
    # Show status tracking
    print("\n3️⃣  Simulating failure tracking...")
    original_index = classifier.current_key_index
    classifier.key_failure_count[original_index] = 2
    
    status = classifier.get_api_key_status()
    print(f"   Key #{original_index + 1} failure count: {status['failure_counts'][f'key_{original_index + 1}']}")
    
    # Test actual classification (optional - requires valid key)
    print("\n4️⃣  Testing actual classification...")
    try:
        prompt = """Classify: 
Title: Machine Learning Overview
Paths: 1. CS > AI > ML
Reply: Path: [exact path] | Confidence: [High]"""
        
        print(f"   Sending request with key #{classifier.current_key_index + 1}...")
        response = classifier.classify(prompt, max_tokens=50, max_retries=2)
        print(f"   ✓ Response: {response[:100]}...")
        
        # Show final status
        status = classifier.get_api_key_status()
        print(f"\n   Final status:")
        print(f"   • Current key: #{status['current_key_index']}")
        print(f"   • Healthy keys: {status['healthy_keys']}/{status['total_keys']}")
        
    except Exception as e:
        print(f"   ⚠️  Classification test skipped: {str(e)[:100]}")
        print(f"   (This is expected if API keys are not configured)")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\n📖 For full documentation, see: API_KEY_ROTATION.md\n")


if __name__ == "__main__":
    main()
