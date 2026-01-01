# Research Article Classification Fine-tuning - File Index

## 📚 Quick Navigation

### Getting Started
1. **`check_setup.sh`** - Run this first to verify your setup
2. **`QUICKSTART.md`** - Get started in 5 minutes
3. **`README.md`** - Complete documentation

### Core Scripts
- **`prepare_data.py`** - Convert JSON files to training format
- **`train_model.py`** - Fine-tune the LLM model
- **`inference.py`** - Classify articles & evaluate model
- **`run_pipeline.sh`** - Automated end-to-end pipeline

### Utilities
- **`utils.py`** - Helper commands (validate, check, count, etc.)
- **`example_usage.py`** - Example code for using the model

### Configuration
- **`requirements.txt`** - Python package dependencies
- **`config.template.sh`** - Training configuration template

### Documentation
- **`SUMMARY.md`** - Complete overview and summary
- **`QUICKSTART.md`** - Quick start guide
- **`README.md`** - Full documentation

## 🚀 Recommended Workflow

```bash
# 1. Check setup
./check_setup.sh

# 2. Read quick start
cat QUICKSTART.md

# 3. Run pipeline
./run_pipeline.sh phi-2 3 4

# 4. Try interactive mode
python inference.py --model_path ./output/phi-2-classification/final_model \
    --base_model microsoft/phi-2 --mode interactive
```

## 📖 File Descriptions

### `prepare_data.py`
**Purpose:** Convert raw JSON articles into training format
**Input:** `Abderrahmane_final_data/` (235 JSON files)
**Output:** `processed_data/` (train, validation, test sets)
**Usage:** `python prepare_data.py`

**What it does:**
- Loads all articles from JSON files
- Extracts title, abstract, classification
- Formats for instruction-based fine-tuning
- Splits into train/validation/test (80/10/10)
- Saves formatted datasets

**When to use:** Run once before training, or when you update the raw data

---

### `train_model.py`
**Purpose:** Fine-tune LLM on classification task
**Input:** Processed training data
**Output:** Fine-tuned model in `output/`
**Usage:** `python train_model.py --model_name phi-2 --num_train_epochs 3 ...`

**What it does:**
- Loads pre-trained model (Mistral, Llama, Phi-2, etc.)
- Applies LoRA for efficient fine-tuning
- Trains on your research articles
- Saves checkpoints and final model

**Key parameters:**
- `--model_name`: Which model to use (phi-2, mistral-7b, etc.)
- `--num_train_epochs`: How many epochs to train
- `--per_device_train_batch_size`: Batch size (adjust for GPU)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--output_dir`: Where to save the model

**When to use:** After preparing data, to train your classifier

---

### `inference.py`
**Purpose:** Use trained model for classification and evaluation
**Input:** Trained model + articles
**Output:** Classifications and/or evaluation metrics
**Usage:** `python inference.py --model_path ./output/model --mode [evaluate|interactive|classify]`

**Modes:**
1. **evaluate**: Test model accuracy on test set
2. **interactive**: Enter articles manually for classification
3. **classify**: Classify a single article via command line

**What it does:**
- Loads fine-tuned model
- Generates classifications for articles
- Computes accuracy metrics (exact match, hierarchical)
- Saves detailed results

**When to use:** After training, to evaluate or use your model

---

### `run_pipeline.sh`
**Purpose:** Automated end-to-end pipeline
**Input:** Model name, epochs, batch size
**Output:** Trained and evaluated model
**Usage:** `./run_pipeline.sh [model] [epochs] [batch_size]`

**Example:** `./run_pipeline.sh phi-2 3 4`

**What it does:**
1. Prepares data (if needed)
2. Trains model with specified parameters
3. Evaluates on test set
4. Saves all results

**When to use:** Convenient way to run entire pipeline with one command

---

### `utils.py`
**Purpose:** Utility commands for common tasks
**Usage:** `python utils.py [command]`

**Commands:**
- `validate`: Check if setup is complete
- `count`: Count total articles in dataset
- `check`: Check processed data status
- `models`: List trained models
- `distribution`: Show data distribution
- `clean`: Clean output directories

**When to use:** For setup verification, data inspection, and maintenance

---

### `example_usage.py`
**Purpose:** Example code showing how to use the model
**What it does:**
- Shows how to load a trained model
- Demonstrates single and batch classification
- Provides template code for your own applications

**When to use:** Reference when integrating the model into your own code

---

### `check_setup.sh`
**Purpose:** Verify that everything is set up correctly
**Usage:** `./check_setup.sh`

**What it checks:**
- Python installation
- Required packages (PyTorch, Transformers, etc.)
- CUDA/GPU availability
- Data directories
- Disk space

**When to use:** Run first before doing anything else

---

### `requirements.txt`
**Purpose:** List of Python package dependencies
**Usage:** `pip install -r requirements.txt`

**Key packages:**
- `torch`: PyTorch for deep learning
- `transformers`: HuggingFace Transformers
- `peft`: Parameter-Efficient Fine-Tuning (LoRA)
- `datasets`: Data loading utilities
- `bitsandbytes`: Quantization for memory efficiency

**When to use:** Install dependencies before starting

---

### `config.template.sh`
**Purpose:** Template for training configuration
**What it contains:**
- All available training parameters
- Example configurations for different scenarios
- Comments explaining each parameter

**When to use:** Reference when customizing training parameters

---

## 📝 Documentation Files

### `QUICKSTART.md`
- Fastest way to get started
- Step-by-step instructions
- Common scenarios and solutions
- 5-minute guide

### `README.md`
- Complete documentation
- Detailed explanations
- All features and options
- Troubleshooting guide

### `SUMMARY.md`
- High-level overview
- What you have and what it does
- Key features and capabilities
- Quick reference

## 🎯 Which File to Use When?

### First Time Setup
1. `check_setup.sh` - Verify environment
2. `requirements.txt` - Install dependencies
3. `QUICKSTART.md` - Read the guide

### Training Your Model
1. `prepare_data.py` - Format your data
2. `train_model.py` - Train the model
   OR
   `run_pipeline.sh` - Do both automatically

### Using Your Model
1. `inference.py --mode evaluate` - Check accuracy
2. `inference.py --mode interactive` - Try it out
3. `example_usage.py` - See code examples

### Maintenance
1. `utils.py validate` - Check setup
2. `utils.py check` - Check data
3. `utils.py models` - List models
4. `utils.py clean` - Clean outputs

## 💡 Pro Tips

1. **Always start with** `check_setup.sh`
2. **Read** `QUICKSTART.md` before anything else
3. **Test with** `tinyllama` before using larger models
4. **Use** `run_pipeline.sh` for convenience
5. **Refer to** `README.md` for details
6. **Check** `example_usage.py` for integration code

## 🔗 File Dependencies

```
Data Flow:
JSON files → prepare_data.py → processed_data/ → train_model.py → output/ → inference.py

Configuration:
config.template.sh → train_model.py
requirements.txt → pip install

Documentation:
QUICKSTART.md (start here)
  ↓
README.md (full details)
  ↓
SUMMARY.md (overview)
```

## ✅ Checklist

- [ ] Ran `check_setup.sh` ✓
- [ ] Read `QUICKSTART.md` ✓
- [ ] Installed dependencies (`pip install -r requirements.txt`) ✓
- [ ] Prepared data (`python prepare_data.py`) ✓
- [ ] Trained model (`./run_pipeline.sh` or `train_model.py`) ✓
- [ ] Evaluated model (`inference.py --mode evaluate`) ✓
- [ ] Tried interactive mode (`inference.py --mode interactive`) ✓

---

**Need help? Check the documentation files or run `python utils.py validate`**
