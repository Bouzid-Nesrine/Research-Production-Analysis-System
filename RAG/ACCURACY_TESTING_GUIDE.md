# RAG Accuracy Testing - Instructions

## Overview
Test the RAG classification system with your 20 labeled articles and calculate accuracy metrics.

## Quick Start

### Step 1: Prepare Your Test Data

Edit `test_data_20_articles.json` and add your 20 labeled articles in this format:

```json
[
  {
    "title": "Deep Learning for Medical Image Segmentation",
    "abstract": "This paper presents a novel deep learning approach...",
    "ground_truth": "Natural Science > Computer and Information Science > Artificial Intelligence > Machine Learning > Deep Learning"
  },
  {
    "title": "Climate Change Impact on Agricultural Productivity",
    "abstract": "We analyze the effects of climate change on crop yields...",
    "ground_truth": "Natural Science > Environmental Science > Climate Science > Climate Change > Climate Impact"
  }
]
```

### Step 2: Run Evaluation

```bash
python evaluate_accuracy.py
```

## What Gets Calculated

### 1. Exact Match Accuracy
Percentage of predictions that exactly match the ground truth path.

### 2. Hierarchical Accuracy
- **Level 1 (Domain)**: Matches at domain level
- **Level 2 (Field)**: Matches up to field level  
- **Level 3 (Subfield)**: Matches up to subfield level

### 3. Performance Metrics
- Average inference time per article
- Total classification time
- Confidence distribution (High/Medium/Low)

### 4. Detailed Results
- Lists all misclassifications
- Shows predicted vs ground truth for each article
- Saves results to JSON and CSV files

## Output Files

### Generated Files:
- `results/accuracy_test_TIMESTAMP.json` - Detailed results with all data
- `results/accuracy_test_TIMESTAMP.csv` - Summary in CSV format

### Example Output:

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
```

## Tips for Labeling Your Data

### 1. Use Exact Taxonomy Paths
Make sure your `ground_truth` paths exactly match paths in your taxonomy:
- Correct: `Natural Science > Computer and Information Science > ...`
- Wrong: `Natural Sciences > Computer Science > ...` (slight differences matter!)

### 2. Check Path Validity
Verify your paths exist in the taxonomy by searching:
```python
from taxonomy_parser import TaxonomyParser
parser = TaxonomyParser("../Taxonomy Building/preprocessed_taxonomy.json")
paths = parser.extract_all_paths()
# Search for your paths in this list
```

### 3. Include Diverse Examples
For robust evaluation, include articles from:
- Different domains (Natural Science, Social Sciences, etc.)
- Different levels of specificity
- Edge cases and ambiguous topics

### 4. Quality Over Quantity
20 well-labeled articles > 100 hastily labeled ones
- Double-check each ground truth path
- Ensure abstracts are representative
- Verify paths are complete (Domain > Field > Subfield > ...)

## Common Issues

### Issue: "Test data file not found"
**Solution**: Create `test_data_20_articles.json` in the RAG directory

### Issue: "Path not found in taxonomy"
**Solution**: Verify your ground_truth paths match exact taxonomy paths (check spelling, capitalization, separators)

### Issue: Low accuracy
**Possible causes:**
- Ground truth labels incorrect
- Articles too ambiguous
- Retrieval not finding correct paths (check top_k parameter)
- LLM needs better prompting

### Issue: API quota errors
**Solution**: 
- Wait for quota reset (usually 1 minute)
- Use multiple API keys (see API_KEY_ROTATION.md)
- Add delays between requests if needed

## Advanced: Custom Evaluation

You can modify `evaluate_accuracy.py` to add custom metrics:

```python
# Example: Calculate top-3 accuracy
def calculate_top3_accuracy(results, pipeline):
    """Check if ground truth is in top 3 retrieved paths"""
    correct = 0
    for r in results:
        retrieved = r['retrieved_paths'][:3]
        if r['ground_truth'] in retrieved:
            correct += 1
    return correct / len(results) * 100
```

## Example Test Data Format

```json
[
  {
    "title": "Quantum Entanglement in Superconducting Qubits",
    "abstract": "We demonstrate quantum entanglement between superconducting qubits using a novel coupling mechanism. Our results show improved coherence times and gate fidelities compared to previous implementations.",
    "ground_truth": "Natural Science > Physics > Quantum Physics > Quantum Computing > Quantum Information"
  },
  {
    "title": "Machine Learning for Drug Discovery",
    "abstract": "This paper applies deep neural networks to predict molecular properties for drug candidate screening. We achieve 92% accuracy on standard benchmarks.",
    "ground_truth": "Natural Science > Computer and Information Science > Artificial Intelligence > Machine Learning > Applications"
  },
  {
    "title": "Economic Impact of COVID-19 Pandemic",
    "abstract": "An econometric analysis of GDP decline across 50 countries during the pandemic, examining fiscal policy responses and recovery trajectories.",
    "ground_truth": "Social Sciences > Economics > Macroeconomics > Economic Growth > Economic Crisis"
  }
]
```

## Interpreting Results

### Good Results (Target):
- Exact Match: 75-85%
- Domain Accuracy: 90-95%
- Field Accuracy: 85-90%
- Average Time: 5-10 seconds

### If Results Are Low:
1. **Check ground truth quality** - Are your labels correct?
2. **Review misclassifications** - Are they understandable errors?
3. **Examine retrieved paths** - Is the correct path being retrieved?
4. **Check confidence** - Low confidence → ambiguous articles
5. **Test with different top_k** - Try retrieving 10 instead of 5 paths

### If Results Are Very High (>95%):
- Great! Your test set might be too easy - try harder examples
- Verify labels are truly correct
- Test with more diverse articles

## Next Steps After Evaluation

1. **Analyze Errors**: Look at misclassifications to understand failure modes
2. **Tune Parameters**: Adjust top_k, temperature if needed
3. **Improve Retrieval**: Update embedding model or descriptions
4. **Expand Test Set**: Add more diverse examples
5. **Production Testing**: Test on real unlabeled data

## Getting Help

Run test with detailed logging:
```bash
python evaluate_accuracy.py 2>&1 | tee evaluation_log.txt
```

Check what paths were retrieved:
- Results show `retrieved_paths` for each article
- Compare with ground truth to see if retrieval is the issue
