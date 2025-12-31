# Testing RAG with 20 Labeled Articles - Quick Start

## 🎯 What You Need to Do

### 1. Add Your 20 Labeled Articles

Edit the file: **`test_data_20_articles.json`**

Replace the example data with your 20 articles in this exact format:

```json
[
  {
    "title": "Your article title here",
    "abstract": "Your article abstract here (full text)",
    "ground_truth": "Domain > Field > Subfield > Specialty > Topic"
  },
  {
    "title": "Second article title",
    "abstract": "Second article abstract...",
    "ground_truth": "Domain > Field > Subfield > ..."
  }
  ... (18 more articles)
]
```

### 2. Run the Evaluation

```bash
cd RAG
python evaluate_accuracy.py
```

### 3. Check Results

The script will show:
- ✅ Exact match accuracy (%)
- ✅ Domain/Field/Subfield accuracy
- ✅ Average inference time
- ✅ Misclassification details
- ✅ Saves results to `results/accuracy_test_TIMESTAMP.json`

## 📋 Important Notes

### Ground Truth Format
Your `ground_truth` paths MUST exactly match taxonomy paths:

**Correct Format:**
```
"Natural Science > Computer and Information Science > Artificial Intelligence > Machine Learning > Deep Learning"
```

**Wrong Format:**
```
"Natural Sciences > Computer Science > AI > ML > Deep Learning"  ❌ (Different wording)
"Natural Science>Computer Science>AI"  ❌ (No spaces around >)
```

### Article Requirements
Each article needs:
- ✅ **title**: The article title
- ✅ **abstract**: Full or substantial abstract (the more text, the better)
- ✅ **ground_truth**: Complete taxonomy path (Domain > Field > Subfield > Specialty > Topic)

## 📊 What Metrics Are Calculated

### 1. Exact Match Accuracy
```
(Number of exact matches / Total articles) × 100
```
Example: 17/20 = 85%

### 2. Hierarchical Accuracy

**Level 1 (Domain):**
- Correct if first level matches
- Example: "Natural Science" vs "Natural Science" ✓

**Level 2 (Field):**  
- Correct if first 2 levels match
- Example: "Natural Science > Physics" ✓

**Level 3 (Subfield):**
- Correct if first 3 levels match
- Example: "Natural Science > Physics > Quantum Physics" ✓

### 3. Confidence Distribution
- How many predictions had High/Medium/Low confidence
- High confidence predictions are usually more accurate

### 4. Performance Metrics
- Average time per article
- Total classification time

## 📁 Output Files

After running, you'll get:

1. **Console Output** - Immediate results display
2. **`results/accuracy_test_TIMESTAMP.json`** - Detailed results with all data
3. **`results/accuracy_test_TIMESTAMP.csv`** - Easy to open in Excel

## Example Output

```
EVALUATION RESULTS
====================================================================

📊 CLASSIFICATION SUMMARY:
  Total Articles: 20
  Successful: 20
  Failed: 0

🎯 ACCURACY METRICS:
  Exact Match Accuracy: 85.00% (17/20)
  Domain Accuracy (Level 1): 95.00%
  Field Accuracy (Level 2): 90.00%
  Subfield Accuracy (Level 3): 85.00%

⏱️  PERFORMANCE:
  Average Inference Time: 6.50s
  Total Time: 130.00s

💪 CONFIDENCE DISTRIBUTION:
  High: 16 (80.0%)
  Medium: 3 (15.0%)
  Low: 1 (5.0%)

❌ MISCLASSIFICATIONS (3):
  Article: Climate Change Impact on Agricultural...
  Ground Truth: Natural Science > Environmental Science > ...
  Predicted:    Natural Science > Agriculture > Climate Impact
  Confidence:   Medium
```

## 🔧 Troubleshooting

### "Test data file not found"
Create `test_data_20_articles.json` in the RAG folder

### "API quota exceeded"
- Wait 60 seconds
- Multiple API keys will automatically rotate
- Check `.env` has `GOOGLE_API_KEYS` configured

### Low Accuracy
1. **Check your ground_truth labels** - Are they correct?
2. **Verify paths exist** - Do they exactly match taxonomy?
3. **Look at misclassifications** - Are errors reasonable?
4. **Check retrieved paths** - Is correct path being retrieved?

### Slow Performance
- Normal: 5-10 seconds per article
- 20 articles ≈ 2-3 minutes total
- With 2 API keys: automatic rotation handles limits

## 💡 Tips for Good Results

### 1. Quality Labels
- Use exact taxonomy paths
- Double-check spelling and formatting
- Verify paths exist in your taxonomy

### 2. Representative Abstracts
- Use full abstracts (not just titles)
- More text = better classification
- Include key domain-specific terms

### 3. Diverse Test Set
Include articles from:
- Different domains
- Different complexity levels
- Different path depths

### 4. Baseline Expectations
- **75-85%** exact match = Good
- **90-95%** domain accuracy = Very good
- **<70%** exact match = Check labels or system

## 🚀 Next Steps

1. **Prepare your data** in `test_data_20_articles.json`
2. **Run evaluation**: `python evaluate_accuracy.py`
3. **Analyze results** - Look at misclassifications
4. **Iterate** - Improve labels or system as needed
5. **Document** - Save results for your report

## 📖 Need More Help?

- **Detailed Guide**: Read `ACCURACY_TESTING_GUIDE.md`
- **API Keys**: See `API_KEY_ROTATION.md`
- **General Docs**: See `README.md`

---

## Quick Checklist

- [ ] Created/edited `test_data_20_articles.json`
- [ ] Added all 20 articles with title, abstract, ground_truth
- [ ] Verified ground_truth paths match taxonomy format
- [ ] Configured API keys in `.env`
- [ ] Ran `python evaluate_accuracy.py`
- [ ] Checked results in console and saved files
- [ ] Analyzed misclassifications
- [ ] Documented accuracy for your report

**Ready to test!** 🎉
