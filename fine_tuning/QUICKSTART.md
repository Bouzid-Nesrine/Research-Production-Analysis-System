# 🚀 Quick Start Guide - Research Article Classification

This guide will help you get started with fine-tuning in minutes.

## ⚡ Super Quick Start (One Command)

```bash
# Install dependencies, prepare data, train, and evaluate
./run_pipeline.sh phi-2 3 4
```

That's it! This will:
1. ✅ Prepare your data from JSON files
2. ✅ Fine-tune Phi-2 model (2.7B params) for 3 epochs
3. ✅ Evaluate on test set
4. ✅ Save results to `./output/phi-2-classification/`

## 📋 Step-by-Step (Manual Control)

### Step 1: Install Dependencies (5 minutes)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Prepare Data (2-5 minutes)

```bash
python prepare_data.py
```

**What this does:**
- Loads 235 JSON files with research articles
- Extracts title, abstract, and classification path
- Formats for instruction-based fine-tuning
- Splits into train (80%), validation (10%), test (10%)
- Saves to `processed_data/` folder

**Output:**
```
Total articles loaded: ~XX,XXX
Formatted: ~XX,XXX articles
Training: ~XX,XXX samples
Validation: ~X,XXX samples
Test: ~X,XXX samples
```

### Step 3: Fine-tune Model (30 min - 2 hours)

#### Option A: Quick Test (Recommended First)
```bash
# Train on small model for 1 epoch (~15 minutes on RTX 3090)
python train_model.py \
    --model_name tinyllama \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --output_dir ./output/test-run
```

#### Option B: Production Training
```bash
# Train Phi-2 for 3 epochs (~45 minutes on RTX 3090)
python train_model.py \
    --model_name phi-2 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --output_dir ./output/phi-2-classification
```

#### Option C: Best Quality (Long Training)
```bash
# Train Mistral-7B for 5 epochs (~2 hours on RTX 3090)
python train_model.py \
    --model_name mistral-7b \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --output_dir ./output/mistral-7b-classification
```

**What to watch:**
- Loss should decrease steadily
- Validation accuracy should increase
- If loss plateaus, training is done

### Step 4: Evaluate Model (5 minutes)

```bash
python inference.py \
    --model_path ./output/phi-2-classification/final_model \
    --base_model microsoft/phi-2 \
    --mode evaluate \
    --test_data ./processed_data/test_instruction.json \
    --output_file ./results.json
```

**Expected Results:**
```
Exact Match Accuracy: 0.55-0.70
Level 1 Accuracy: 0.85-0.95
Level 2 Accuracy: 0.75-0.85
Level 3 Accuracy: 0.65-0.75
```

### Step 5: Try It Out! (Interactive)

```bash
python inference.py \
    --model_path ./output/phi-2-classification/final_model \
    --base_model microsoft/phi-2 \
    --mode interactive
```

Then enter:
```
Title: Graphene-based nanomaterials for energy storage
Abstract: We present a novel approach to synthesize graphene-based composite materials with enhanced electrochemical properties...
```

Get instant classification! 🎉

## 🎯 Choosing the Right Model

| Model | Parameters | Speed | Quality | GPU Memory | Use Case |
|-------|-----------|-------|---------|------------|----------|
| **TinyLlama** | 1.1B | ⚡⚡⚡⚡ | ⭐⭐⭐ | 4GB | Quick testing |
| **Gemma-2B** | 2B | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 6GB | Good balance |
| **Phi-2** | 2.7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8GB | **Recommended** |
| **Llama-2-7B** | 7B | ⚡⚡ | ⭐⭐⭐⭐⭐ | 16GB | Best quality |
| **Mistral-7B** | 7B | ⚡⚡ | ⭐⭐⭐⭐⭐ | 16GB | Best quality |

**Recommendation:** Start with **Phi-2** - great balance of speed and quality!

## 💾 GPU Memory Requirements

| Configuration | GPU Memory | Training Time* |
|--------------|------------|----------------|
| TinyLlama, batch=8, 4-bit | 4GB | ~20 min |
| Phi-2, batch=4, 4-bit | 8GB | ~45 min |
| Mistral-7B, batch=2, 4-bit | 12GB | ~90 min |
| Mistral-7B, batch=4, 4-bit | 16GB | ~60 min |

*For 3 epochs on ~50K samples, RTX 3090

## 🔥 Common Issues & Solutions

### ❌ Out of Memory Error

**Solution 1:** Reduce batch size
```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
```

**Solution 2:** Use smaller model
```bash
--model_name phi-2  # instead of mistral-7b
```

**Solution 3:** Reduce sequence length
```bash
--max_length 256  # instead of 512
```

### ❌ Training Too Slow

**Solution 1:** Use smaller model
```bash
--model_name tinyllama
```

**Solution 2:** Increase batch size (if memory allows)
```bash
--per_device_train_batch_size 8
```

**Solution 3:** Train on subset
```bash
--data_path ./processed_data/sample_instruction.json
```

### ❌ Poor Accuracy

**Solution 1:** Train longer
```bash
--num_train_epochs 5  # instead of 3
```

**Solution 2:** Use larger model
```bash
--model_name mistral-7b  # instead of phi-2
```

**Solution 3:** Adjust learning rate
```bash
--learning_rate 1e-4  # try different values: 5e-5, 1e-4, 2e-4, 5e-4
```

## 📊 Monitoring Training

### Option 1: Terminal Output
Watch the loss and learning rate in real-time:
```
Step 100: loss=2.145, lr=0.0002
Step 200: loss=1.823, lr=0.00019
Step 300: loss=1.541, lr=0.00018
```

### Option 2: TensorBoard
```bash
# In a separate terminal
tensorboard --logdir ./output/phi-2-classification
```
Then open http://localhost:6006

### Option 3: Weights & Biases
```bash
# Enable in training command
--report_to wandb
```

## 🎓 What Each File Does

- **`prepare_data.py`** - Converts your JSON articles into training format
- **`train_model.py`** - Fine-tunes the LLM on your data
- **`inference.py`** - Use trained model to classify new articles
- **`run_pipeline.sh`** - Runs everything automatically
- **`requirements.txt`** - Python packages needed

## 📝 Typical Workflow

```bash
# 1. First time setup
pip install -r requirements.txt
python prepare_data.py

# 2. Quick test run (15 min)
python train_model.py --model_name tinyllama --num_train_epochs 1 --output_dir ./test

# 3. Check if it works
python inference.py --model_path ./test/final_model --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode interactive

# 4. If happy, train full model (45 min)
python train_model.py --model_name phi-2 --num_train_epochs 3 --output_dir ./final

# 5. Evaluate
python inference.py --model_path ./final/final_model --base_model microsoft/phi-2 --mode evaluate

# 6. Use it!
python inference.py --model_path ./final/final_model --base_model microsoft/phi-2 --mode interactive
```

## 🚀 Next Steps

1. ✅ Run the quick test with TinyLlama
2. ✅ If it works, train Phi-2 for production
3. ✅ Evaluate and check accuracy
4. ✅ If accuracy is good, you're done!
5. ✅ If accuracy needs improvement, try Mistral-7B

## 💡 Pro Tips

1. **Start small:** Always test with TinyLlama or small data first
2. **Monitor GPU:** Use `nvidia-smi` to watch memory usage
3. **Save checkpoints:** Training crashes? Resume from last checkpoint
4. **Experiment:** Try different learning rates and models
5. **Data quality matters:** Clean, accurate labels = better model

## 🎉 Success Looks Like

After training, you should be able to:
- ✅ Enter any research title and abstract
- ✅ Get accurate taxonomy classification
- ✅ See hierarchical accuracy > 85% at top levels
- ✅ Process hundreds of articles quickly

**Good luck! 🚀**
