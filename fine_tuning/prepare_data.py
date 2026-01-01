"""
Data Preparation Script for Research Article Classification Fine-tuning
This script processes JSON files containing research articles and prepares them
for fine-tuning a language model to classify articles based on title and abstract.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import random
from sklearn.model_selection import train_test_split
from collections import Counter


def load_all_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all JSON files from the data directory.
    
    Args:
        data_dir: Path to directory containing JSON files
        
    Returns:
        List of all articles from all JSON files
    """
    all_articles = []
    json_files = list(Path(data_dir).glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                all_articles.extend(articles)
                print(f"Loaded {len(articles)} articles from {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    print(f"\nTotal articles loaded: {len(all_articles)}")
    return all_articles


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()


def format_for_instruction_tuning(articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Format articles for instruction-based fine-tuning (Alpaca/Llama style).
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        List of formatted training examples
    """
    formatted_data = []
    skipped = 0
    
    for article in articles:
        title = article.get('title', '') or article.get('display_name', '')
        abstract = article.get('abstract', '')
        classification = article.get('classification_path', '')
        
        # Skip if missing critical information
        if not title or not abstract or not classification:
            skipped += 1
            continue
        
        # Clean the text
        title = clean_text(title)
        abstract = clean_text(abstract)
        classification = clean_text(classification)
        
        # Format as instruction-following task
        instruction = "Classify the following research article into its appropriate taxonomy path based on the title and abstract. Provide the full classification path from field to subfield."
        
        input_text = f"Title: {title}\n\nAbstract: {abstract}"
        
        output_text = classification
        
        formatted_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    
    print(f"Formatted {len(formatted_data)} articles")
    print(f"Skipped {skipped} articles due to missing data")
    
    return formatted_data


def format_for_causal_lm(articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Format articles for causal language model fine-tuning.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        List of formatted training examples with text field
    """
    formatted_data = []
    skipped = 0
    
    for article in articles:
        title = article.get('title', '') or article.get('display_name', '')
        abstract = article.get('abstract', '')
        classification = article.get('classification_path', '')
        
        # Skip if missing critical information
        if not title or not abstract or not classification:
            skipped += 1
            continue
        
        # Clean the text
        title = clean_text(title)
        abstract = clean_text(abstract)
        classification = clean_text(classification)
        
        # Format as a complete text with special tokens
        text = (
            f"<|system|>\nYou are a research article classifier. Classify articles into their appropriate taxonomy.\n"
            f"<|user|>\nClassify this article:\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"<|assistant|>\n"
            f"{classification}<|end|>"
        )
        
        formatted_data.append({
            "text": text
        })
    
    print(f"Formatted {len(formatted_data)} articles")
    print(f"Skipped {skipped} articles due to missing data")
    
    return formatted_data


def analyze_dataset(data: List[Dict[str, Any]]):
    """Print statistics about the dataset."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Get all classification paths
    classifications = [item.get('classification_path', '') for item in data]
    
    # Count unique classifications
    unique_classifications = set(classifications)
    print(f"\nTotal samples: {len(data)}")
    print(f"Unique classification paths: {len(unique_classifications)}")
    
    # Show distribution
    classification_counts = Counter(classifications)
    print(f"\nTop 10 most common classifications:")
    for clf, count in classification_counts.most_common(10):
        print(f"  {clf}: {count}")
    
    # Analyze taxonomy levels
    levels = [len(clf.split(' > ')) for clf in classifications if clf]
    if levels:
        print(f"\nTaxonomy depth statistics:")
        print(f"  Min levels: {min(levels)}")
        print(f"  Max levels: {max(levels)}")
        print(f"  Average levels: {sum(levels)/len(levels):.2f}")
    
    print("="*60 + "\n")


def save_datasets(data: List[Dict[str, str]], output_dir: str, format_type: str = "instruction"):
    """
    Split data into train/validation/test sets and save them.
    
    Args:
        data: Formatted training data
        output_dir: Directory to save the datasets
        format_type: Type of formatting ('instruction' or 'causal')
    """
    # Shuffle data
    random.shuffle(data)
    
    # Split: 80% train, 10% validation, 10% test
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    datasets = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    for split_name, split_data in datasets.items():
        output_path = os.path.join(output_dir, f"{split_name}_{format_type}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {split_name} set to {output_path}")
    
    # Save a small sample for quick testing
    sample_data = train_data[:100]
    sample_path = os.path.join(output_dir, f"sample_{format_type}.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    print(f"Saved sample set (100 examples) to {sample_path}")


def main():
    """Main execution function."""
    # Configuration
    DATA_DIR = "./Abderrahmane_final_data/Abderrahmane_final_data"
    OUTPUT_DIR = "./processed_data"
    
    print("="*60)
    print("RESEARCH ARTICLE CLASSIFICATION - DATA PREPARATION")
    print("="*60)
    
    # Load all data
    print("\n[1/4] Loading data...")
    all_articles = load_all_data(DATA_DIR)
    
    if not all_articles:
        print("Error: No articles loaded. Please check the data directory.")
        return
    
    # Analyze raw data
    print("\n[2/4] Analyzing dataset...")
    analyze_dataset(all_articles)
    
    # Format data in both styles
    print("\n[3/4] Formatting data...")
    
    # Format 1: Instruction-tuning format (for models like Llama, Mistral)
    print("\nFormatting for instruction-tuning...")
    instruction_data = format_for_instruction_tuning(all_articles)
    
    # Format 2: Causal LM format (alternative approach)
    print("\nFormatting for causal language modeling...")
    causal_data = format_for_causal_lm(all_articles)
    
    # Save datasets
    print("\n[4/4] Saving datasets...")
    save_datasets(instruction_data, OUTPUT_DIR, "instruction")
    save_datasets(causal_data, OUTPUT_DIR, "causal")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nProcessed data saved to: {OUTPUT_DIR}")
    print("\nYou can now proceed with fine-tuning using:")
    print("  - train_instruction.json for instruction-based fine-tuning")
    print("  - train_causal.json for causal LM fine-tuning")
    print("\nRecommended models (< 7B parameters):")
    print("  - mistralai/Mistral-7B-v0.1")
    print("  - meta-llama/Llama-2-7b-hf")
    print("  - google/gemma-7b")
    print("  - microsoft/phi-2 (2.7B)")
    print("  - TinyLlama/TinyLlama-1.1B-Chat-v1.0")


if __name__ == "__main__":
    main()
