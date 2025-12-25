#!/bin/bash

# Research Article Classification - Training Pipeline
# This script runs the complete fine-tuning pipeline

set -e  # Exit on error

echo "=========================================="
echo "Research Article Classification Pipeline"
echo "=========================================="
echo ""

# Configuration
MODEL=${1:-"phi-2"}
EPOCHS=${2:-3}
BATCH_SIZE=${3:-4}
OUTPUT_DIR="./output/${MODEL}-classification"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output: $OUTPUT_DIR"
echo ""

# Step 1: Prepare data (if not already done)
if [ ! -d "./processed_data" ]; then
    echo "[1/3] Preparing data..."
    python prepare_data.py
    echo "✓ Data preparation complete"
    echo ""
else
    echo "[1/3] Data already prepared (./processed_data exists)"
    echo ""
fi

# Step 2: Fine-tune model
echo "[2/3] Fine-tuning model..."
python train_model.py \
    --model_name "$MODEL" \
    --data_path ./processed_data/train_instruction.json \
    --val_data_path ./processed_data/validation_instruction.json \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --bf16 True \
    --gradient_checkpointing True \
    --max_length 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --report_to none

echo "✓ Training complete"
echo ""

# Step 3: Evaluate model
echo "[3/3] Evaluating model..."

# Determine base model name
case $MODEL in
    "mistral-7b")
        BASE_MODEL="mistralai/Mistral-7B-v0.1"
        ;;
    "llama2-7b")
        BASE_MODEL="meta-llama/Llama-2-7b-hf"
        ;;
    "phi-2")
        BASE_MODEL="microsoft/phi-2"
        ;;
    "tinyllama")
        BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ;;
    "gemma-2b")
        BASE_MODEL="google/gemma-2b"
        ;;
    *)
        BASE_MODEL=""
        ;;
esac

if [ -n "$BASE_MODEL" ]; then
    python inference.py \
        --model_path "$OUTPUT_DIR/final_model" \
        --base_model "$BASE_MODEL" \
        --mode evaluate \
        --test_data ./processed_data/test_instruction.json \
        --output_file "$OUTPUT_DIR/evaluation_results.json"
else
    python inference.py \
        --model_path "$OUTPUT_DIR/final_model" \
        --mode evaluate \
        --test_data ./processed_data/test_instruction.json \
        --output_file "$OUTPUT_DIR/evaluation_results.json"
fi

echo "✓ Evaluation complete"
echo ""

echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  - View evaluation results: cat $OUTPUT_DIR/evaluation_results.json"
echo "  - Try interactive mode: python inference.py --model_path $OUTPUT_DIR/final_model --base_model $BASE_MODEL --mode interactive"
echo ""
