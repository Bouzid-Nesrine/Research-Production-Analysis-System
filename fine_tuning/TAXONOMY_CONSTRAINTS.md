# Fine-tuning with Taxonomy Constraints - Important Update!

## 🎯 Key Change: Taxonomy-Constrained Classification

The model is now trained and validated against your **actual taxonomy tree** from `final_combined_taxonomy.json`. This ensures the LLM only outputs valid classification paths that exist in your taxonomy.

## 📋 Why This Matters

**Before:** Model could generate arbitrary classification paths
**Now:** Model is constrained to only valid paths from your taxonomy

### Benefits:
- ✅ **Guaranteed valid outputs** - All predictions match your taxonomy structure
- ✅ **Auto-correction** - Invalid predictions are automatically corrected to closest match
- ✅ **Validation metrics** - Track how often model generates valid vs invalid paths
- ✅ **Taxonomy compliance** - Ensures consistency with your research classification system

## 🚀 Updated Workflow

### 1. Data Preparation (with Taxonomy)

```bash
python prepare_data_with_taxonomy.py
```

**What this does:**
- Loads your taxonomy from `../Taxonomy Building/final_combined_taxonomy.json`
- Extracts all valid classification paths (~1000+ paths)
- Validates your training data against the taxonomy
- Creates `taxonomy_reference.json` for inference
- Formats training data with taxonomy constraints

**Output:**
```
processed_data/
├── train_instruction.json
├── validation_instruction.json
├── test_instruction.json
├── sample_instruction.json
└── taxonomy_reference.json  ← New! Contains all valid paths
```

### 2. Training (Same as Before)

```bash
python train_model.py \
    --model_name phi-2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --output_dir ./output/phi2-taxonomy-classifier
```

The model learns from examples that only contain valid taxonomy paths.

### 3. Inference with Validation

```bash
python inference_with_taxonomy.py \
    --model_path ./output/phi2-taxonomy-classifier/final_model \
    --base_model microsoft/phi-2 \
    --taxonomy_path ./processed_data/taxonomy_reference.json \
    --mode evaluate
```

**New Features:**
- ✅ Validates each prediction against taxonomy
- ✅ Auto-corrects invalid predictions using fuzzy matching
- ✅ Reports validation statistics
- ✅ Shows which predictions were corrected

## 📊 Example Output

```
Taxonomy Validation:
  Valid predictions: 847/1000 (84.7%)
  Auto-corrected: 153 (15.3%)

Exact Match Accuracy: 0.6789
Hierarchical Accuracy:
  Level 1: 0.9234
  Level 2: 0.8456
  Level 3: 0.7123
```

## 🔍 Your Taxonomy Structure

From `final_combined_taxonomy.json`:

```
Natural Science
├── Mathematics (1.01)
│   ├── Pure Mathematics
│   │   ├── Algebra
│   │   ├── Analysis
│   │   └── ...
│   ├── Applied Mathematics
│   └── Statistics and Probability
├── Computer and Information Science (1.02)
├── Physical Science (1.03)
└── ...

Engineering and Technology
├── Civil Engineering (2.01)
├── Electrical and Information Engineering (2.02)
├── Material Engineering (2.05)
│   └── Material Science
│       ├── Metal and Alloy
│       ├── Ceramic
│       ├── Polymer
│       ├── Composite
│       └── Nanomaterial  ← Your articles classify here
└── ...

(7 major domains, 1000+ valid paths)
```

## 🛠️ Complete Pipeline

### Option 1: Quick Start

```bash
# 1. Prepare data with taxonomy
python prepare_data_with_taxonomy.py

# 2. Train model
python train_model.py \
    --model_name phi-2 \
    --num_train_epochs 3 \
    --output_dir ./output/phi2-classifier

# 3. Evaluate with taxonomy validation
python inference_with_taxonomy.py \
    --model_path ./output/phi2-classifier/final_model \
    --base_model microsoft/phi-2 \
    --mode evaluate
```

### Option 2: Automated Script

I'll create an automated script that runs everything:

```bash
./run_taxonomy_pipeline.sh phi-2 3 4
```

## 📈 Understanding the Results

### Validation Metrics

1. **Valid Predictions**: Percentage of predictions that exactly match taxonomy paths
2. **Auto-corrected**: Predictions that were invalid but corrected using fuzzy matching
3. **Exact Match Accuracy**: Predictions that match the true label exactly
4. **Hierarchical Accuracy**: Accuracy at each level of the taxonomy tree

### Example Corrections

```
Predicted: "Engineering and Technology > Material Science > Nanomaterial"
Corrected: "Engineering and Technology > Material Engineering > Material Science > Nanomaterial"
         ↑ Missing level auto-corrected
```

## 🎯 Classification Example

**Input:**
```
Title: "Graphene-based nanocomposites for energy storage"
Abstract: "We synthesized graphene oxide composites with enhanced electrochemical properties..."
```

**Model Output (Validated):**
```
Classification: Engineering and Technology > Material Engineering > Material Science > Nanomaterial
✓ Valid taxonomy path
✓ Matches training data format
```

## 💡 Key Files

| File | Purpose |
|------|---------|
| `prepare_data_with_taxonomy.py` | **Use this** - Prepares data with taxonomy validation |
| `inference_with_taxonomy.py` | **Use this** - Inference with auto-correction |
| `taxonomy_reference.json` | Contains all valid paths (auto-generated) |
| `prepare_data.py` | Old version (no taxonomy constraints) |
| `inference.py` | Old version (no validation) |

## ⚠️ Important Notes

1. **Always use the taxonomy-aware scripts** (`*_with_taxonomy.py`)
2. **The taxonomy reference must match your training data** - regenerate if taxonomy changes
3. **Auto-correction helps with minor formatting** - but train longer if many corrections needed
4. **Invalid paths in training data are skipped** - check warnings during data preparation

## 🔧 Troubleshooting

### Many invalid predictions?
- Check if model needs more training epochs
- Verify training data matches taxonomy format
- Consider using larger model (mistral-7b instead of phi-2)

### Auto-corrections not working?
- Predictions may be too different from valid paths
- Check `taxonomy_reference.json` contains expected paths
- May need to adjust fuzzy matching threshold in code

### Training data skipped?
- Some article classifications don't match taxonomy
- Check warnings during `prepare_data_with_taxonomy.py`
- Verify `classification_path` field format in your JSON files

## 📚 Next Steps

1. Run `prepare_data_with_taxonomy.py` to create taxonomy-constrained training data
2. Train your model as usual
3. Use `inference_with_taxonomy.py` for validated predictions
4. Check validation metrics to ensure taxonomy compliance

---

**Remember:** Your taxonomy from `final_combined_taxonomy.json` is now the source of truth for all classifications! 🎯
