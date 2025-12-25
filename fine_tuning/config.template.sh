# Training Configuration Template
# Copy this file and modify for your training runs

# Model Configuration
MODEL_NAME="phi-2"  # Options: mistral-7b, llama2-7b, phi-2, tinyllama, gemma-2b
USE_4BIT=true
USE_8BIT=false

# Data Configuration
DATA_PATH="./processed_data/train_instruction.json"
VAL_DATA_PATH="./processed_data/validation_instruction.json"
MAX_LENGTH=512

# Training Hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=2e-4
WARMUP_RATIO=0.03
LR_SCHEDULER="cosine"

# LoRA Configuration
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

# Output Configuration
OUTPUT_DIR="./output/model-classification"
LOGGING_STEPS=10
SAVE_STRATEGY="epoch"
EVAL_STRATEGY="epoch"

# Optimization
BF16=true
TF32=true
GRADIENT_CHECKPOINTING=true

# Hardware Configuration
# For single GPU: "0"
# For multi-GPU: "0,1,2,3"
CUDA_VISIBLE_DEVICES="0"

# Weights & Biases (optional)
# Set to "wandb" to enable logging, "none" to disable
REPORT_TO="none"
# WANDB_PROJECT="research-classification"
# WANDB_RUN_NAME="phi-2-classification-run1"

# ========================================
# Example Configurations for Different Scenarios
# ========================================

# === SCENARIO 1: Quick Testing (Small Model, Few Epochs) ===
# MODEL_NAME="tinyllama"
# NUM_EPOCHS=1
# BATCH_SIZE=8
# DATA_PATH="./processed_data/sample_instruction.json"

# === SCENARIO 2: Memory Constrained (< 8GB GPU) ===
# MODEL_NAME="phi-2"
# USE_4BIT=true
# BATCH_SIZE=1
# GRADIENT_ACCUMULATION=16
# MAX_LENGTH=256

# === SCENARIO 3: High Performance (24GB+ GPU) ===
# MODEL_NAME="mistral-7b"
# BATCH_SIZE=8
# GRADIENT_ACCUMULATION=2
# MAX_LENGTH=1024
# LORA_R=32

# === SCENARIO 4: Best Quality (Long Training) ===
# MODEL_NAME="mistral-7b"
# NUM_EPOCHS=10
# BATCH_SIZE=4
# LORA_R=32
# LORA_ALPHA=64
