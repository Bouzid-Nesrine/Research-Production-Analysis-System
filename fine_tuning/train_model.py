"""
Fine-tuning Script for Research Article Classification
Supports multiple models (< 7B parameters) with LoRA/QLoRA for efficient training
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb


# Model configurations (all < 7B parameters)
MODEL_CONFIGS = {
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral 7B - Strong performance, 7B params"
    },
    "llama2-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "description": "Llama 2 7B - Meta's flagship, 7B params"
    },
    "phi-2": {
        "name": "microsoft/phi-2",
        "description": "Phi-2 - Microsoft's efficient model, 2.7B params"
    },
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama - Very fast training, 1.1B params"
    },
    "gemma-2b": {
        "name": "google/gemma-2b",
        "description": "Gemma 2B - Google's efficient model, 2B params"
    }
}


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name: str = field(
        default="mistral-7b",
        metadata={"help": f"Model to use. Options: {', '.join(MODEL_CONFIGS.keys())}"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Use 4-bit quantization (QLoRA)"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit quantization"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    data_path: str = field(
        default="./processed_data/train_instruction.json",
        metadata={"help": "Path to training data"}
    )
    val_data_path: str = field(
        default="./processed_data/validation_instruction.json",
        metadata={"help": "Path to validation data"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )


def load_model_and_tokenizer(model_args: ModelArguments):
    """
    Load model and tokenizer with optional quantization.
    
    Args:
        model_args: Model configuration arguments
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = MODEL_CONFIGS[model_args.model_name]
    model_name = model_config["name"]
    
    print(f"\n{'='*60}")
    print(f"Loading model: {model_config['description']}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization
    quantization_config = None
    if model_args.use_4bit:
        print("Using 4-bit quantization (QLoRA)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif model_args.use_8bit:
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if not quantization_config else None,
    )
    
    # Prepare for k-bit training if using quantization
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def create_lora_model(model, lora_args: LoraArguments):
    """
    Apply LoRA to the model.
    
    Args:
        model: Base model
        lora_args: LoRA configuration
        
    Returns:
        Model with LoRA applied
    """
    target_modules = lora_args.lora_target_modules.split(",")
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable%: {100 * trainable_params / total_params:.2f}%\n")
    
    return model


def prepare_dataset(data_path: str, tokenizer, max_length: int):
    """
    Load and prepare dataset for training.
    
    Args:
        data_path: Path to JSON data file
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Processed dataset
    """
    # Load JSON data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        """Tokenize the data."""
        # Format: instruction + input + output
        prompts = []
        for instruction, input_text, output in zip(
            examples['instruction'],
            examples['input'],
            examples['output']
        ):
            prompt = f"{instruction}\n\n{input_text}\n\nClassification: {output}"
            prompts.append(prompt)
        
        # Tokenize
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Set labels (copy of input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def main():
    """Main training function."""
    # Parse arguments
    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        LoraArguments,
        TrainingArguments
    ))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    print("\n" + "="*60)
    print("RESEARCH ARTICLE CLASSIFICATION - FINE-TUNING")
    print("="*60)
    
    # Print configuration
    print(f"\nModel: {model_args.model_name}")
    print(f"Training data: {data_args.data_path}")
    print(f"Output directory: {training_args.output_dir}")
    
    # Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # Apply LoRA
    print("\n[2/5] Applying LoRA...")
    model = create_lora_model(model, lora_args)
    
    # Prepare datasets
    print("\n[3/5] Preparing datasets...")
    train_dataset = prepare_dataset(data_args.data_path, tokenizer, data_args.max_length)
    eval_dataset = prepare_dataset(data_args.val_data_path, tokenizer, data_args.max_length)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize trainer
    print("\n[4/5] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n[5/5] Starting training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    final_output_dir = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"\nModel saved to: {final_output_dir}")
    print("\nYou can now use the model for inference with inference.py")


if __name__ == "__main__":
    main()
