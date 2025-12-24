# API Setup Guide

## Alibaba Cloud Qwen 2.5 API Configuration

This project uses **Alibaba Cloud's DashScope API** for Qwen 2.5 instead of loading the model locally. This provides several advantages:
- ✅ No GPU required
- ✅ No 28GB model download
- ✅ Faster inference
- ✅ Pay-per-use pricing

## Getting Your API Key

### 1. Create Alibaba Cloud Account
Visit: https://www.alibabacloud.com/

### 2. Access DashScope Console
Go to: https://dashscope.console.aliyun.com/

### 3. Generate API Key
- Navigate to the API Key management page
- Click "Create API Key"
- Copy your API key (keep it secure!)

## Configuration

### 1. Create `.env` File
Copy the example file and add your API key:

```bash
cp .env.example .env
```

### 2. Edit `.env` File
Open `.env` and add your API key:

```env
ALIBABA_API_KEY=sk-your-actual-api-key-here
```

**⚠️ Important:** Never commit your `.env` file to version control!

## Available Models

The API supports several Qwen 2.5 models:

| Model | Speed | Cost | Use Case |
|-------|-------|------|----------|
| `qwen-turbo` | Fast | Lowest | High-volume classification |
| `qwen-plus` | Medium | Medium | **Recommended for this project** |
| `qwen-max` | Slower | Highest | Maximum accuracy |

### Changing the Model

Edit [config.py](config.py):

```python
LLM_MODEL_NAME = "qwen-plus"  # Change to qwen-turbo or qwen-max
```

## Usage

### Python Code

```python
from llm_classifier import LLMClassifier

# Initialize (reads API key from .env)
classifier = LLMClassifier(model_name="qwen-plus")

# Or pass API key directly
classifier = LLMClassifier(
    api_key="sk-your-api-key",
    model_name="qwen-plus"
)

# Classify article
result = classifier.classify_article(
    title="Your article title",
    abstract="Your article abstract",
    relevant_paths=["Path 1", "Path 2", ...]
)
```

### Full Pipeline

```python
from rag_pipeline import RAGClassificationPipeline

# Initialize (automatically uses API)
pipeline = RAGClassificationPipeline(auto_setup=True)

# Classify
result = pipeline.classify_article(
    title="Your article title",
    abstract="Your article abstract"
)
```

## API Parameters

You can customize the API behavior in [config.py](config.py):

```python
RAG_CONFIG = {
    "temperature": 0.3,      # 0-2, lower = more deterministic
    "max_tokens": 256,       # Maximum response length
    "top_p": 0.9,           # Nucleus sampling (0-1)
}
```

## Testing API Connection

Test your API setup:

```python
from llm_classifier import LLMClassifier
import logging

logging.basicConfig(level=logging.INFO)

# Test initialization
try:
    classifier = LLMClassifier(model_name="qwen-plus")
    print("✓ API connection successful!")
except ValueError as e:
    print(f"✗ API setup failed: {e}")
```

## Troubleshooting

### "API key required" Error
- Check that `.env` file exists in the RAG directory
- Verify `ALIBABA_API_KEY` is set in `.env`
- Ensure no extra spaces or quotes around the API key

### "Authorization failed" Error
- Verify your API key is correct
- Check that your Alibaba Cloud account is active
- Ensure DashScope API is enabled in your account

### "Request timeout" Error
- Check your internet connection
- Increase timeout in config.py:
  ```python
  LLM_API_CONFIG = {
      "timeout": 120,  # Increase from 60
  }
  ```

### Rate Limiting
If you hit API rate limits:
- Reduce batch size in config.py
- Add delays between requests
- Consider upgrading your API plan

## Cost Management

### Estimating Costs
Typical usage per article:
- Input: ~500-1000 tokens (article + retrieved paths)
- Output: ~100-200 tokens (classification response)
- Total: ~600-1200 tokens per article

Check current pricing at: https://dashscope.console.aliyun.com/billing

### Optimization Tips
1. **Use qwen-turbo** for large-scale classification
2. **Reduce top_k** to retrieve fewer paths (saves input tokens)
3. **Lower max_tokens** if responses are verbose
4. **Batch processing** to maximize throughput

## Security Best Practices

1. **Never commit `.env` file** - already in .gitignore
2. **Rotate API keys** periodically
3. **Use environment variables** in production
4. **Monitor API usage** in DashScope console
5. **Set up billing alerts** to prevent unexpected charges

## Next Steps

After configuring your API:

1. ✅ Run retrieval tests (no API needed):
   ```bash
   python test_2_vector_database.py
   python test_3_embeddings.py
   python test_4_retrieval.py
   ```

2. ✅ Test LLM classification:
   ```bash
   python quickstart.py
   ```

3. ✅ Process your dataset:
   ```bash
   python -c "from rag_pipeline import RAGClassificationPipeline; pipeline = RAGClassificationPipeline(auto_setup=True); # Your classification code"
   ```

## Support

- **Alibaba Cloud DashScope Docs**: https://help.aliyun.com/zh/dashscope/
- **API Reference**: https://help.aliyun.com/zh/dashscope/developer-reference/api-details
- **Community Forum**: https://developer.aliyun.com/ask/
