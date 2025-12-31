# API Key Rotation - Quick Reference Card

## 🚀 Quick Start (30 seconds)

### 1. Add API Keys to .env
```bash
# Edit RAG/.env and add (comma-separated, NO spaces):
GOOGLE_API_KEYS=AIzaSy...key1,AIzaSy...key2,AIzaSy...key3,AIzaSy...key4
```

### 2. Use Normally (No Code Changes!)
```python
from rag_pipeline import RAGClassificationPipeline

pipeline = RAGClassificationPipeline(auto_setup=True)
result = pipeline.classify_article(title, abstract)
# ✨ Automatic rotation happens behind the scenes!
```

### 3. Test It
```bash
python demo_api_rotation.py
```

---

## 📊 Status Monitoring

```python
status = classifier.get_api_key_status()

# Example output:
{
    'total_keys': 4,              # Total keys loaded
    'current_key_index': 2,       # Currently using key #2
    'healthy_keys': 4,            # Keys with <5 failures
    'failure_counts': {           # Failures per key
        'key_1': 0,
        'key_2': 1,
        'key_3': 0,
        'key_4': 0
    }
}
```

---

## 🔄 How Rotation Works

```
Request → API Key #1 → Success ✓
Request → API Key #1 → Success ✓
Request → API Key #1 → Rate Limit! 
       → Switch to Key #2 → Success ✓
Request → API Key #2 → Success ✓
```

**Automatic triggers:**
- "rate limit exceeded"
- "quota exceeded"
- "resource exhausted"
- HTTP 429 errors

---

## 💡 Common Commands

```python
# Check status
status = classifier.get_api_key_status()
print(f"Using key #{status['current_key_index']}")

# Reset failures (if keys recover)
classifier.reset_key_failures()

# Force rotation (testing)
classifier._rotate_api_key()

# Classify with custom retries
result = classifier.classify(prompt, max_retries=5)
```

---

## 🎯 Configuration

| Setting | Where | Default | Notes |
|---------|-------|---------|-------|
| API Keys | `.env` | - | Comma-separated |
| Max Retries | `classify()` | 3 | Per request |
| Failure Threshold | `llm_classifier.py` | 5 | Before skipping key |
| Retry Delay | Auto | 0.5-1s | Between attempts |

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Only one key available" | Check `.env` has `GOOGLE_API_KEYS=key1,key2,key3` (no spaces!) |
| "All keys exhausted" | Wait 1 min, run `classifier.reset_key_failures()` |
| Keys not rotating | Check logs for actual error message |
| Want more throughput | Add more API keys (4-5 recommended) |

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `.env` | Your API keys configuration |
| `API_KEY_ROTATION.md` | Full documentation |
| `demo_api_rotation.py` | Quick demo script |
| `test_api_key_rotation.py` | Test suite |
| `llm_classifier.py` | Implementation code |

---

## ✅ Verification Checklist

- [ ] Added 2+ API keys to `.env`
- [ ] Keys separated by commas (no spaces)
- [ ] Ran `python demo_api_rotation.py`
- [ ] Saw "Loaded X API key(s)" message
- [ ] Classification works normally

---

## 🎉 Benefits at a Glance

✅ **4x throughput** with 4 keys  
✅ **Zero downtime** on rate limits  
✅ **No code changes** required  
✅ **Smart failover** to healthy keys  
✅ **Full monitoring** of all keys  

---

## 📖 Learn More

- **Complete Guide**: `API_KEY_ROTATION.md`
- **Get API Keys**: `API_SETUP.md`
- **Main Docs**: `README.md`

---

**Need help?** Run the test: `python test_api_key_rotation.py`
