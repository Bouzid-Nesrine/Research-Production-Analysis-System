"""
Utility script for common fine-tuning tasks
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter


def count_articles():
    """Count total articles in dataset."""
    data_dir = "./Abderrahmane_final_data/Abderrahmane_final_data"
    json_files = list(Path(data_dir).glob("*.json"))
    
    total = 0
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            total += len(articles)
    
    print(f"Total JSON files: {len(json_files)}")
    print(f"Total articles: {total:,}")


def check_processed_data():
    """Check if processed data exists and show statistics."""
    processed_dir = "./processed_data"
    
    if not os.path.exists(processed_dir):
        print("❌ Processed data not found!")
        print("Run: python prepare_data.py")
        return
    
    print("✅ Processed data found!\n")
    
    files = {
        "Training": "train_instruction.json",
        "Validation": "validation_instruction.json",
        "Test": "test_instruction.json",
        "Sample": "sample_instruction.json"
    }
    
    for name, filename in files.items():
        filepath = os.path.join(processed_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"{name:12s}: {len(data):>6,} samples")
        else:
            print(f"{name:12s}: ❌ Not found")


def list_models():
    """List all trained models."""
    output_dir = "./output"
    
    if not os.path.exists(output_dir):
        print("❌ No trained models found!")
        print("Output directory doesn't exist: ./output")
        return
    
    models = []
    for item in os.listdir(output_dir):
        model_path = os.path.join(output_dir, item)
        if os.path.isdir(model_path):
            final_model = os.path.join(model_path, "final_model")
            if os.path.exists(final_model):
                models.append({
                    'name': item,
                    'path': final_model,
                    'size': get_dir_size(final_model)
                })
    
    if not models:
        print("❌ No trained models found!")
        print("Train a model first: python train_model.py ...")
        return
    
    print(f"✅ Found {len(models)} trained model(s):\n")
    for model in models:
        print(f"  📦 {model['name']}")
        print(f"     Path: {model['path']}")
        print(f"     Size: {model['size']:.2f} GB")
        print()


def get_dir_size(path):
    """Get directory size in GB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total / (1024**3)  # Convert to GB


def show_data_distribution():
    """Show distribution of classifications in processed data."""
    data_file = "./processed_data/train_instruction.json"
    
    if not os.path.exists(data_file):
        print("❌ Training data not found!")
        print("Run: python prepare_data.py")
        return
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    classifications = [item['output'] for item in data]
    
    # Count by top-level field
    top_level = [c.split(' > ')[0] if ' > ' in c else c for c in classifications]
    top_level_counts = Counter(top_level)
    
    print(f"Total samples: {len(data):,}\n")
    print("Top-level field distribution:")
    for field, count in top_level_counts.most_common():
        percentage = (count / len(data)) * 100
        print(f"  {field:50s}: {count:>6,} ({percentage:>5.2f}%)")


def validate_setup():
    """Validate that everything is set up correctly."""
    print("🔍 Validating setup...\n")
    
    issues = []
    
    # Check raw data
    data_dir = "./Abderrahmane_final_data/Abderrahmane_final_data"
    if not os.path.exists(data_dir):
        issues.append(f"❌ Raw data directory not found: {data_dir}")
    else:
        json_files = list(Path(data_dir).glob("*.json"))
        if len(json_files) == 0:
            issues.append(f"❌ No JSON files found in: {data_dir}")
        else:
            print(f"✅ Raw data: {len(json_files)} JSON files")
    
    # Check processed data
    processed_dir = "./processed_data"
    if not os.path.exists(processed_dir):
        print("⚠️  Processed data not found (run prepare_data.py)")
    else:
        required_files = ["train_instruction.json", "validation_instruction.json", "test_instruction.json"]
        missing = [f for f in required_files if not os.path.exists(os.path.join(processed_dir, f))]
        if missing:
            issues.append(f"❌ Missing processed files: {', '.join(missing)}")
        else:
            print("✅ Processed data: All files present")
    
    # Check Python packages
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda} (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("⚠️  CUDA: Not available (will use CPU - very slow!)")
    except ImportError:
        issues.append("❌ PyTorch not installed")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        issues.append("❌ Transformers not installed")
    
    try:
        import peft
        print(f"✅ PEFT: {peft.__version__}")
    except ImportError:
        issues.append("❌ PEFT not installed")
    
    # Summary
    print("\n" + "="*60)
    if issues:
        print("❌ SETUP INCOMPLETE\n")
        for issue in issues:
            print(issue)
        print("\nFix these issues and run again.")
    else:
        print("✅ SETUP COMPLETE - Ready to train!")
    print("="*60)


def clean_outputs():
    """Clean output directories."""
    output_dir = "./output"
    
    if not os.path.exists(output_dir):
        print("No output directory to clean.")
        return
    
    print("⚠️  This will delete all trained models!")
    response = input("Are you sure? (yes/no): ")
    
    if response.lower() == 'yes':
        import shutil
        shutil.rmtree(output_dir)
        print("✅ Output directory cleaned.")
    else:
        print("Cancelled.")


def main():
    parser = argparse.ArgumentParser(description="Utility script for fine-tuning tasks")
    parser.add_argument(
        "command",
        choices=["count", "check", "models", "distribution", "validate", "clean"],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    
    if args.command == "count":
        print("COUNTING ARTICLES")
        print("="*60 + "\n")
        count_articles()
    
    elif args.command == "check":
        print("CHECKING PROCESSED DATA")
        print("="*60 + "\n")
        check_processed_data()
    
    elif args.command == "models":
        print("LISTING TRAINED MODELS")
        print("="*60 + "\n")
        list_models()
    
    elif args.command == "distribution":
        print("DATA DISTRIBUTION")
        print("="*60 + "\n")
        show_data_distribution()
    
    elif args.command == "validate":
        print("VALIDATING SETUP")
        print("="*60 + "\n")
        validate_setup()
    
    elif args.command == "clean":
        print("CLEANING OUTPUTS")
        print("="*60 + "\n")
        clean_outputs()
    
    print("="*60)


if __name__ == "__main__":
    main()
