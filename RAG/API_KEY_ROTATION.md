# API Key Rotation - User Guide

## Overview

The RAG classification system now supports **automatic API key rotation** to handle rate limits and quotas when using multiple Google AI Studio API keys. When one key hits its limit, the system automatically switches to the next available key.

## Benefits

✅ **Uninterrupted Processing**: Automatically switches keys when rate limits are hit  
✅ **Higher Throughput**: Effectively multiply your API quota by the number of keys  
✅ **Fault Tolerance**: Continues working even if some keys fail  
✅ **Smart Retry Logic**: Tracks failures and skips problematic keys  
✅ **Zero Configuration**: Works seamlessly with existing code  

## Setup

### Step 1: Get Multiple API Keys

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create multiple API keys (recommended: 4-5 keys)
3. Copy each key

### Step 2: Configure Environment

Edit your `.env` file:

```bash
# Multiple API keys for automatic rotation (comma-separated)
GOOGLE_API_KEYS=AIzaSyBfcYhKW9Hnvl1GqaS5U4bwuDpc6efvTPo,AIzaSyCrFtbVqO49d3KkBVrJE345Uw8oNCLbeME,AIzaSyC...,AIzaSyD...

# Optional: Fallback single key (used if GOOGLE_API_KEYS is not set)
GOOGLE_API_KEY=AIzaSyBfcYhKW9Hnvl1GqaS5U4bwuDpc6efvTPo
```

**Important**: Separate keys with commas (`,`) and no spaces.

### Step 3: Verify Setup

Run the test script:

```bash
python test_api_key_rotation.py
```

Expected output:
```
API KEY CONFIGURATION TEST
✓ Classifier initialized successfully
  Total API keys loaded: 4
  Current key index: 1
  Healthy keys: 4/4
```

## How It Works

### Automatic Key Rotation

```
Request 1 → API Key #1 → Success ✓
Request 2 → API Key #1 → Success ✓
Request 3 → API Key #1 → Rate Limit! → Rotate to Key #2 → Success ✓
Request 4 → API Key #2 → Success ✓
Request 5 → API Key #2 → Rate Limit! → Rotate to Key #3 → Success ✓
...
```

### Error Detection

The system automatically detects rate limit errors:
- "rate limit exceeded"
- "quota exceeded"
- "resource exhausted"
- HTTP 429 errors

When detected, it:
1. Logs the error
2. Rotates to next key
3. Retries the request
4. Tracks failure count per key

### Failure Tracking

- Each key tracks consecutive failures
- Keys with 5+ failures are skipped during rotation
- Reset failures with: `classifier.reset_key_failures()`

## Usage Examples

### Basic Usage (No Code Changes)

```python
from rag_pipeline import RAGClassificationPipeline

# Initialize pipeline - automatically uses multiple keys from .env
pipeline = RAGClassificationPipeline(auto_setup=True)

# Classify - automatic rotation happens behind the scenes
result = pipeline.classify_article(
    title="Deep Learning for Computer Vision",
    abstract="This paper presents..."
)
```

### Manual Key Management

```python
from llm_classifier import LLMClassifier

# Option 1: Load from environment (recommended)
classifier = LLMClassifier()

# Option 2: Provide keys programmatically
api_keys = ["key1", "key2", "key3", "key4"]
classifier = LLMClassifier(api_keys=api_keys)

# Check status
status = classifier.get_api_key_status()
print(f"Using key #{status['current_key_index']}")
print(f"Healthy keys: {status['healthy_keys']}/{status['total_keys']}")

# Reset failure counts if needed
classifier.reset_key_failures()
```

### Batch Processing

```python
# Process large batches without worrying about rate limits
articles = [...]  # 1000s of articles

results = pipeline.batch_classify(
    articles, 
    show_progress=True
)

# System automatically rotates keys as needed
# Check final status
status = pipeline.llm_classifier.get_api_key_status()
print(f"Final status: {status['failure_counts']}")
```

## Configuration

### Retry Settings

Modify retry behavior in classification:

```python
response = classifier.classify(
    prompt,
    max_retries=5,  # Try up to 5 times (default: 3)
    temperature=0.1
)
```

### Key Failure Threshold

Keys are skipped after 5 consecutive failures. To change this, modify in `llm_classifier.py`:

```python
# In _rotate_api_key method
if self.key_failure_count[self.current_key_index] < 10:  # Changed from 5
    ...
```

## Monitoring

### Check Current Status

```python
status = classifier.get_api_key_status()

print(f"Total keys: {status['total_keys']}")
print(f"Current key: #{status['current_key_index']}")
print(f"Healthy keys: {status['healthy_keys']}")
print(f"Failures per key: {status['failure_counts']}")
```

### Example Output

```python
{
    'total_keys': 4,
    'current_key_index': 2,
    'current_key_prefix': 'AIzaSyBfcYhKW9Hnvl1G...',
    'failure_counts': {
        'key_1': 0,
        'key_2': 1,
        'key_3': 0,
        'key_4': 0
    },
    'healthy_keys': 4
}
```

### Logs

The system logs all key rotations:

```
2025-12-31 10:15:23 - WARNING - API key #1 hit rate limit: quota exceeded
2025-12-31 10:15:23 - INFO - Rotated to API key #2
2025-12-31 10:15:24 - INFO - Retrying with new API key (attempt 2/3)
```

## Best Practices

### 1. Use 4-5 API Keys
- Google AI Studio allows multiple keys per account
- More keys = higher effective quota
- Recommended: 4-5 keys for production use

### 2. Monitor Failure Counts
```python
# Check status every 100 classifications
if count % 100 == 0:
    status = classifier.get_api_key_status()
    if status['healthy_keys'] < 2:
        logger.warning("Low healthy keys - consider adding more")
```

### 3. Reset Failures Periodically
```python
# Reset daily or after long breaks
classifier.reset_key_failures()
```

### 4. Handle Edge Cases
```python
try:
    result = pipeline.classify_article(title, abstract)
except Exception as e:
    if "all keys exhausted" in str(e).lower():
        logger.error("All API keys are rate limited - wait and retry")
        time.sleep(300)  # Wait 5 minutes
        classifier.reset_key_failures()
    else:
        raise
```

## Troubleshooting

### Issue: "Only one API key available"

**Cause**: `GOOGLE_API_KEYS` not set or has single key

**Solution**: 
```bash
# Check .env file
GOOGLE_API_KEYS=key1,key2,key3,key4  # Comma-separated, no spaces
```

### Issue: "All API keys have failed multiple times"

**Cause**: All keys exceeded quota or have persistent errors

**Solutions**:
1. Wait for quota to reset (usually 1 minute for rate limits)
2. Reset failure counts: `classifier.reset_key_failures()`
3. Check if keys are valid: test in AI Studio console
4. Add more API keys

### Issue: Keys not rotating

**Cause**: Error is not recognized as rate limit

**Solution**: Check logs for actual error message and add to detection in `llm_classifier.py`:

```python
is_rate_limit = any(keyword in error_msg for keyword in 
    ['rate limit', 'quota', 'resource exhausted', '429', 
     'your_custom_error_message'])  # Add custom errors here
```

## Performance Impact

- **Key Rotation Overhead**: ~0.5 seconds (minimal)
- **Additional Memory**: ~negligible (~1KB per key)
- **Throughput Increase**: ~4x with 4 keys (linear scaling)

## Migration from Single Key

Existing code works without changes:

```python
# Old code (still works)
pipeline = RAGClassificationPipeline()
result = pipeline.classify_article(title, abstract)

# Automatically uses multiple keys if GOOGLE_API_KEYS is set
```

## Security Notes

⚠️ **Keep your API keys secure**:
- Never commit `.env` to version control
- Use `.gitignore` to exclude `.env`
- Rotate keys if accidentally exposed
- Consider using environment variables in production

## Testing

Test the rotation functionality:

```bash
# Run automated tests
python test_api_key_rotation.py

# Test with real classification
python -c "
from llm_classifier import LLMClassifier
classifier = LLMClassifier()
print(classifier.get_api_key_status())
"
```

## Support

For issues or questions:
1. Check logs in `logs/rag_classification.log`
2. Run test script: `python test_api_key_rotation.py`
3. Verify `.env` configuration
4. Check API key validity in Google AI Studio
