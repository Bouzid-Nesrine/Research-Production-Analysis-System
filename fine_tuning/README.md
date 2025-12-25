# Research Article Classification Fine-tuning

A complete pipeline for fine-tuning Large Language Models (< 7B parameters) to classify research articles into taxonomies based on their titles and abstracts.

## 📋 Overview

This project fine-tunes pre-trained language models to automatically classify research articles into hierarchical taxonomy paths (e.g., "Engineering and Technology > Material Engineering > Material Science > Nanomaterial").

**Key Features:**
- ✅ Support for multiple models (Mistral-7B, Llama-2-7B, Phi-2, TinyLlama, Gemma-2B)
- ✅ Efficient training with LoRA/QLoRA (4-bit quantization)
- ✅ Automatic data preparation and formatting
- ✅ Interactive inference mode
- ✅ Comprehensive evaluation metrics
- ✅ Hierarchical classification accuracy

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Process raw JSON files into training format
python prepare_data.py
```

This will:
- Load all articles from `Abderrahmane_final_data/`
- Format data for instruction-based fine-tuning
- Split into train/validation/test sets (80/10/10)
- Save to `processed_data/` directory

**Output files:**
- `train_instruction.json` - Training set
- `validation_instruction.json` - Validation set
- `test_instruction.json` - Test set
- `sample_instruction.json` - Small sample for testing

### 3. Fine-tune Model

#### Quick Training (Recommended for Testing)

```bash
python train_model.py \
    --model_name phi-2 \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --bf16 True
```

#### Full Training Configuration

```bash
python train_model.py \
    --model_name mistral-7b \
    --data_path ./processed_data/train_instruction.json \
    --val_data_path ./processed_data/validation_instruction.json \
    --output_dir ./output/mistral-7b-classification \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --max_length 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
```

**Available Models:**
- `mistral-7b` - Mistral 7B (7B params, strong performance)
- `llama2-7b` - Llama 2 7B (7B params, Meta's flagship)
- `phi-2` - Phi-2 (2.7B params, efficient)
- `tinyllama` - TinyLlama (1.1B params, fastest)
- `gemma-2b` - Gemma 2B (2B params, Google)

### 4. Evaluate Model

```bash
# Evaluate on test set
python inference.py \
    --model_path ./output/mistral-7b-classification/final_model \
    --base_model mistralai/Mistral-7B-v0.1 \
    --mode evaluate \
    --test_data ./processed_data/test_instruction.json \
    --output_file ./evaluation_results.json
```

**Metrics calculated:**
- Exact match accuracy
- Hierarchical accuracy (at each taxonomy level)
- Detailed per-sample results

### 5. Interactive Classification

```bash
# Start interactive mode
python inference.py \
    --model_path ./output/mistral-7b-classification/final_model \
    --base_model mistralai/Mistral-7B-v0.1 \
    --mode interactive
```

Then enter article titles and abstracts to get classifications in real-time.

### 6. Single Article Classification

```bash
python inference.py \
    --model_path ./output/mistral-7b-classification/final_model \
    --base_model mistralai/Mistral-7B-v0.1 \
    --mode classify \
    --title "2D Homologous Perovskites as Light-Absorbing Materials" \
    --abstract "We report on the fabrication and properties of 2D perovskite thin films..."
```

## 📊 Dataset Statistics

After running `prepare_data.py`, you'll see statistics like:
- Total articles
- Unique classification paths
- Distribution across categories
- Taxonomy depth information

## 🔧 Advanced Configuration

### Memory Optimization

For limited GPU memory, use these settings:

```bash
python train_model.py \
    --model_name phi-2 \
    --use_4bit True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --max_length 256
```

### LoRA Parameters

Adjust LoRA configuration for different trade-offs:

```bash
# Higher rank = better quality, more memory
--lora_r 32 --lora_alpha 64

# Lower rank = less memory, faster training
--lora_r 8 --lora_alpha 16
```

### Training on Subset

For quick testing, use the sample dataset:

```bash
python train_model.py \
    --data_path ./processed_data/sample_instruction.json \
    --val_data_path ./processed_data/sample_instruction.json \
    --num_train_epochs 1
```

## 📁 Project Structure

```
fine_tuning/
├── prepare_data.py              # Data preparation script
├── train_model.py               # Fine-tuning script
├── inference.py                 # Inference & evaluation
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── Abderrahmane_final_data/     # Raw data (235 JSON files)
├── processed_data/              # Formatted training data (created by prepare_data.py)
└── output/                      # Model checkpoints (created during training)
```

## 🎯 Expected Results

With proper training, you should achieve:
- **Level 1 (Field)**: 85-95% accuracy
- **Level 2 (Subfield)**: 75-85% accuracy
- **Level 3**: 65-75% accuracy
- **Level 4 (Full path)**: 55-70% accuracy

## 💡 Tips & Best Practices

1. **Start Small**: Test with `phi-2` or `tinyllama` first
2. **Monitor Training**: Check loss curves and validation accuracy
3. **Experiment with Hyperparameters**: 
   - Learning rate: Try 1e-4 to 5e-4
   - Epochs: Start with 3-5
   - Batch size: Adjust based on GPU memory
4. **Data Quality**: Ensure abstracts and classifications are clean
5. **Use GPU**: Training on CPU will be extremely slow

## 🐛 Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use `--use_4bit True`
- Reduce `--max_length`
- Use smaller model (phi-2 or tinyllama)

### Slow Training
- Increase batch size if memory allows
- Enable `--tf32 True` on Ampere GPUs
- Use `--bf16 True` instead of fp32
- Reduce dataset size for testing

### Poor Performance
- Train for more epochs
- Increase model size
- Tune learning rate
- Check data quality

## 📝 License

This project is for research and educational purposes.

## 🤝 Contributing

Feel free to improve the code, add features, or report issues!

## 📧 Contact

For questions or issues, please create an issue in the repository.

---

**Happy Fine-tuning! 🚀**
