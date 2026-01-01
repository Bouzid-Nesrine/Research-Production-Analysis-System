#!/bin/bash

# Taxonomy-Constrained Classification Pipeline
# This script uses your actual taxonomy tree to constrain model outputs

set -e

echo "=========================================="
echo "Taxonomy-Constrained Classification"
echo "=========================================="
echo ""

# Configuration
MODEL=${1:-"phi-2"}
EPOCHS=${2:-3}
BATCH_SIZE=${3:-4}
OUTPUT_DIR="./output/${MODEL}-taxonomy-classifier"
TAXONOMY_PATH="../Taxonomy Building/final_combined_taxonomy.json"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output: $OUTPUT_DIR"
echo "  Taxonomy: $TAXONOMY_PATH"
echo ""

# Check if taxonomy file exists
if [ ! -f "$TAXONOMY_PATH" ]; then
    echo "❌ Error: Taxonomy file not found at $TAXONOMY_PATH"
    echo "Please ensure final_combined_taxonomy.json exists in ../Taxonomy Building/"
    exit 1
fi

# Step 1: Prepare data with taxonomy constraints
if [ ! -d "./processed_data" ] || [ ! -f "./processed_data/taxonomy_reference.json" ]; then
    echo "[1/3] Preparing data with taxonomy constraints..."
    python prepare_data_with_taxonomy.py
    echo "✓ Data preparation complete"
    echo ""
else
    echo "[1/3] Data already prepared with taxonomy"
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

# Step 3: Evaluate with taxonomy validation
echo "[3/3] Evaluating model with taxonomy validation..."

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
    python inference_with_taxonomy.py \
        --model_path "$OUTPUT_DIR/final_model" \
        --base_model "$BASE_MODEL" \
        --taxonomy_path ./processed_data/taxonomy_reference.json \
        --mode evaluate \
        --test_data ./processed_data/test_instruction.json \
        --output_file "$OUTPUT_DIR/evaluation_results_validated.json"
else
    python inference_with_taxonomy.py \
        --model_path "$OUTPUT_DIR/final_model" \
        --taxonomy_path ./processed_data/taxonomy_reference.json \
        --mode evaluate \
        --test_data ./processed_data/test_instruction.json \
        --output_file "$OUTPUT_DIR/evaluation_results_validated.json"
fi

echo "✓ Evaluation complete"
echo ""

echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "✅ Model trained with taxonomy constraints"
echo "✅ Predictions validated against your taxonomy"
echo "✅ Auto-correction enabled for invalid paths"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - $OUTPUT_DIR/final_model/ (trained model)"
echo "  - $OUTPUT_DIR/evaluation_results_validated.json (results)"
echo "  - ./processed_data/taxonomy_reference.json (valid paths)"
echo ""
echo "Validation metrics:"
echo "  Check evaluation output above for:"
echo "  - Valid prediction rate"
echo "  - Auto-correction rate"
echo "  - Hierarchical accuracy at each level"
echo ""
echo "Next steps:"
echo "  - Review validation metrics"
echo "  - Check auto-correction rate (should be < 20%)"
echo "  - If many corrections needed, train longer or use larger model"
echo ""
