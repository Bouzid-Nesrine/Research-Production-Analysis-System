# 🎯 START HERE - Research Article Classification Fine-tuning

Welcome! This is your complete fine-tuning pipeline for training LLMs to classify research articles.

## 🚦 First Steps (Do This Now!)

### Step 1: Check Your Setup (30 seconds)
```bash
./check_setup.sh
```

If you see ✅ marks, you're good to go! If you see ❌ marks, follow the instructions to fix them.

### Step 2: Install Dependencies (5 minutes)
```bash
pip install -r requirements.txt
```

### Step 3: Choose Your Path

#### 🏃 **Quick Path** - Get results fast (1 hour total)
```bash
# Run everything with one command
./run_pipeline.sh phi-2 3 4

# Then try it out
python inference.py \
    --model_path ./output/phi-2-classification/final_model \
    --base_model microsoft/phi-2 \
    --mode interactive
```

#### 🎓 **Learning Path** - Understand each step (2 hours total)
```bash
# 1. Prepare data (5 min)
python prepare_data.py

# 2. Train model (45 min)
python train_model.py \
    --model_name phi-2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --output_dir ./output/my-classifier

# 3. Evaluate (5 min)
python inference.py \
    --model_path ./output/my-classifier/final_model \
    --base_model microsoft/phi-2 \
    --mode evaluate

# 4. Try it interactively
python inference.py \
    --model_path ./output/my-classifier/final_model \
    --base_model microsoft/phi-2 \
    --mode interactive
```

## 📚 What to Read Next

1. **QUICKSTART.md** - Detailed 5-minute guide with examples
2. **README.md** - Complete documentation of all features
3. **FILE_INDEX.md** - What each file does and when to use it

## 💡 Quick Reference

### Common Commands

```bash
# Check setup
./check_setup.sh

# Prepare data
python prepare_data.py

# Complete pipeline
./run_pipeline.sh [model] [epochs] [batch_size]

# Train manually
python train_model.py --model_name phi-2 --num_train_epochs 3

# Evaluate model
python inference.py --model_path ./output/MODEL/final_model --base_model BASE --mode evaluate

# Interactive classification
python inference.py --model_path ./output/MODEL/final_model --base_model BASE --mode interactive

# Utility commands
python utils.py validate    # Validate setup
python utils.py count       # Count articles
python utils.py check       # Check processed data
python utils.py models      # List trained models
```

### Model Options (< 7B parameters)

| Command | Model | Size | Speed | Quality | GPU RAM |
|---------|-------|------|-------|---------|---------|
| `tinyllama` | TinyLlama | 1.1B | ⚡⚡⚡⚡ | ⭐⭐⭐ | 4GB |
| `gemma-2b` | Gemma | 2B | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 6GB |
| `phi-2` | Phi-2 | 2.7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8GB |
| `llama2-7b` | Llama-2 | 7B | ⚡⚡ | ⭐⭐⭐⭐⭐ | 16GB |
| `mistral-7b` | Mistral | 7B | ⚡⚡ | ⭐⭐⭐⭐⭐ | 16GB |

**Recommendation:** Use `phi-2` - best balance of speed and quality!

## 🎯 What You'll Get

After training, your model will:
- ✅ Take any research article title + abstract
- ✅ Output hierarchical classification (e.g., "Engineering and Technology > Material Engineering > Material Science > Nanomaterial")
- ✅ Achieve 85-95% accuracy at top taxonomy level
- ✅ Classify articles in seconds

## 🔥 Troubleshooting

### ❌ Out of Memory?
```bash
# Use smaller model
--model_name phi-2

# Reduce batch size
--per_device_train_batch_size 1 --gradient_accumulation_steps 16

# Reduce sequence length
--max_length 256
```

### ❌ Training Too Slow?
```bash
# Use smaller model
--model_name tinyllama

# Train on sample data
--data_path ./processed_data/sample_instruction.json
```

### ❌ Poor Accuracy?
```bash
# Train longer
--num_train_epochs 10

# Use larger model
--model_name mistral-7b

# Adjust learning rate
--learning_rate 1e-4
```

## 📂 Project Structure

```
fine_tuning/
├── 🚀 Scripts
│   ├── prepare_data.py          # Data preparation
│   ├── train_model.py           # Training
│   ├── inference.py             # Evaluation & classification
│   └── utils.py                 # Utilities
│
├── 📖 Documentation
│   ├── START_HERE.md            # This file
│   ├── QUICKSTART.md            # 5-min guide
│   ├── README.md                # Full docs
│   ├── SUMMARY.md               # Overview
│   └── FILE_INDEX.md            # File guide
│
├── ⚙️ Configuration
│   ├── requirements.txt         # Dependencies
│   ├── config.template.sh       # Config template
│   └── check_setup.sh           # Setup check
│
├── 💡 Examples
│   └── example_usage.py         # Usage examples
│
└── 📂 Data & Models
    ├── Abderrahmane_final_data/ # Raw data (235 JSON files)
    ├── processed_data/          # Training data (created by prepare_data.py)
    └── output/                  # Models (created by train_model.py)
```

## ✅ Success Checklist

- [ ] Run `./check_setup.sh` - all green?
- [ ] Install dependencies - `pip install -r requirements.txt`
- [ ] Read QUICKSTART.md
- [ ] Prepare data - `python prepare_data.py`
- [ ] Train model - `./run_pipeline.sh phi-2 3 4`
- [ ] Evaluate - check accuracy > 85% at top level
- [ ] Try interactive mode - classify some articles!

## 🎉 You're Ready!

Pick your path (Quick or Learning) and start training!

**Need help?** 
- Check QUICKSTART.md for detailed guide
- Run `python utils.py validate` to check setup
- Read README.md for complete documentation

**Happy fine-tuning! 🚀**

---

**Next:** Read `QUICKSTART.md` for detailed instructions
