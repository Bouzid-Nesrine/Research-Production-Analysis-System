"""
Enhanced Data Preparation Script with Taxonomy Tree
This version includes the full taxonomy structure in prompts to constrain LLM outputs
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Set
import random
from sklearn.model_selection import train_test_split
from collections import Counter


def load_taxonomy(taxonomy_path: str) -> Dict:
    """Load the taxonomy structure."""
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_all_paths(taxonomy_dict: Dict, prefix: str = "") -> Set[str]:
    """
    Extract all valid classification paths from taxonomy.
    
    Args:
        taxonomy_dict: Taxonomy dictionary
        prefix: Current path prefix
        
    Returns:
        Set of all valid classification paths
    """
    paths = set()
    
    if isinstance(taxonomy_dict, dict):
        for key, value in taxonomy_dict.items():
            # Clean the key (remove codes like (1.01))
            clean_key = key.split('(')[0].strip() if '(' in key else key
            current_path = f"{prefix} > {clean_key}" if prefix else clean_key
            
            if isinstance(value, dict):
                # Recurse into nested structure
                paths.update(extract_all_paths(value, current_path))
            elif isinstance(value, list):
                # Leaf node - add path for each item
                for item in value:
                    leaf_path = f"{current_path} > {item}"
                    paths.add(leaf_path)
            
            # Also add intermediate path
            paths.add(current_path)
    
    return paths


def format_taxonomy_as_text(taxonomy_dict: Dict, indent: int = 0) -> str:
    """
    Format taxonomy as hierarchical text for inclusion in prompts.
    
    Args:
        taxonomy_dict: Taxonomy dictionary
        indent: Current indentation level
        
    Returns:
        Formatted taxonomy string
    """
    lines = []
    indent_str = "  " * indent
    
    if isinstance(taxonomy_dict, dict):
        for key, value in taxonomy_dict.items():
            # Skip metadata
            if key == "metadata":
                continue
                
            # Clean the key
            clean_key = key.split('(')[0].strip() if '(' in key else key
            lines.append(f"{indent_str}- {clean_key}")
            
            if isinstance(value, dict):
                lines.append(format_taxonomy_as_text(value, indent + 1))
            elif isinstance(value, list) and len(value) > 0:
                for item in value:
                    lines.append(f"{indent_str}  - {item}")
    
    return "\n".join(lines)


def create_taxonomy_summary(valid_paths: Set[str]) -> str:
    """
    Create a concise summary of the taxonomy for prompts.
    
    Args:
        valid_paths: Set of all valid paths
        
    Returns:
        Formatted taxonomy summary
    """
    # Group by top-level domain
    domains = {}
    for path in valid_paths:
        parts = path.split(' > ')
        if len(parts) > 0:
            domain = parts[0]
            if domain not in domains:
                domains[domain] = set()
            domains[domain].add(path)
    
    summary_lines = ["Valid Classification Taxonomy:"]
    for domain in sorted(domains.keys()):
        summary_lines.append(f"\n{domain}:")
        # Show a sample of paths from this domain
        sample_paths = sorted(list(domains[domain]))[:5]
        for path in sample_paths:
            summary_lines.append(f"  • {path}")
        if len(domains[domain]) > 5:
            summary_lines.append(f"  ... and {len(domains[domain]) - 5} more paths")
    
    return "\n".join(summary_lines)


def load_all_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON files from the data directory."""
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
    text = " ".join(text.split())
    return text.strip()


def format_for_instruction_tuning(
    articles: List[Dict[str, Any]],
    valid_paths: Set[str],
    include_full_taxonomy: bool = False
) -> List[Dict[str, str]]:
    """
    Format articles for instruction-based fine-tuning with taxonomy constraints.
    
    Args:
        articles: List of article dictionaries
        valid_paths: Set of valid taxonomy paths
        include_full_taxonomy: Whether to include full taxonomy in each prompt
        
    Returns:
        List of formatted training examples
    """
    formatted_data = []
    skipped = 0
    invalid_paths = 0
    
    # Create taxonomy reference (first 20 examples per domain for context)
    taxonomy_context = create_taxonomy_summary(valid_paths)
    
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
        
        # Validate classification against taxonomy
        if classification not in valid_paths:
            # Try to find closest match (handle minor formatting differences)
            close_matches = [p for p in valid_paths if p.lower() == classification.lower()]
            if close_matches:
                classification = close_matches[0]
            else:
                print(f"Warning: Invalid path '{classification}' - skipping")
                invalid_paths += 1
                continue
        
        # Format instruction with taxonomy constraint
        if include_full_taxonomy:
            instruction = f"""Classify the following research article into its appropriate taxonomy path. You must choose from the valid taxonomy below.

{taxonomy_context}

Provide the full classification path from the taxonomy above."""
        else:
            instruction = """Classify the following research article into its appropriate taxonomy path based on the title and abstract. Choose from the predefined research taxonomy covering: Natural Science, Engineering and Technology, Medical and Health Science, Agricultural Science, Social Science, Humanity and Art, and Interdisciplinary Fields. Provide the full hierarchical path (e.g., "Domain > Field > Subfield > Specialty")."""
        
        input_text = f"Title: {title}\n\nAbstract: {abstract}"
        output_text = classification
        
        formatted_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "valid_classification": True
        })
    
    print(f"Formatted {len(formatted_data)} articles")
    print(f"Skipped {skipped} articles due to missing data")
    print(f"Skipped {invalid_paths} articles due to invalid classification paths")
    
    return formatted_data


def save_taxonomy_reference(valid_paths: Set[str], output_dir: str):
    """Save taxonomy reference for use during inference."""
    taxonomy_ref = {
        "valid_paths": sorted(list(valid_paths)),
        "total_paths": len(valid_paths),
        "domains": {}
    }
    
    # Group by domain
    for path in valid_paths:
        parts = path.split(' > ')
        if len(parts) > 0:
            domain = parts[0]
            if domain not in taxonomy_ref["domains"]:
                taxonomy_ref["domains"][domain] = []
            taxonomy_ref["domains"][domain].append(path)
    
    # Save
    output_path = os.path.join(output_dir, "taxonomy_reference.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomy_ref, f, indent=2, ensure_ascii=False)
    
    print(f"Saved taxonomy reference to {output_path}")
    print(f"Total valid paths: {len(valid_paths)}")
    print(f"Domains: {len(taxonomy_ref['domains'])}")


def save_datasets(data: List[Dict[str, str]], output_dir: str, format_type: str = "instruction"):
    """Split data into train/validation/test sets and save them."""
    random.shuffle(data)
    
    # Split: 80% train, 10% validation, 10% test
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Save small sample
    sample_data = train_data[:100]
    sample_path = os.path.join(output_dir, f"sample_{format_type}.json")
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    print(f"Saved sample set (100 examples) to {sample_path}")


def main():
    """Main execution function."""
    # Configuration
    TAXONOMY_PATH = "../Taxonomy Building/final_combined_taxonomy.json"
    DATA_DIR = "./Abderrahmane_final_data/Abderrahmane_final_data"
    OUTPUT_DIR = "./processed_data"
    
    print("="*60)
    print("RESEARCH ARTICLE CLASSIFICATION - DATA PREPARATION")
    print("WITH TAXONOMY CONSTRAINTS")
    print("="*60)
    
    # Load taxonomy
    print("\n[1/5] Loading taxonomy...")
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    print(f"Taxonomy version: {taxonomy.get('metadata', {}).get('version', 'N/A')}")
    print(f"Total domains: {taxonomy.get('metadata', {}).get('total_domains', 'N/A')}")
    
    # Extract valid paths
    print("\n[2/5] Extracting valid classification paths...")
    valid_paths = extract_all_paths(taxonomy.get('taxonomy', {}))
    print(f"Total valid paths: {len(valid_paths)}")
    
    # Show sample paths
    print("\nSample valid paths:")
    for path in sorted(list(valid_paths))[:5]:
        print(f"  • {path}")
    
    # Load article data
    print("\n[3/5] Loading article data...")
    all_articles = load_all_data(DATA_DIR)
    
    if not all_articles:
        print("Error: No articles loaded. Please check the data directory.")
        return
    
    # Format data with taxonomy constraints
    print("\n[4/5] Formatting data with taxonomy constraints...")
    formatted_data = format_for_instruction_tuning(all_articles, valid_paths)
    
    # Save datasets and taxonomy reference
    print("\n[5/5] Saving datasets and taxonomy reference...")
    save_datasets(formatted_data, OUTPUT_DIR, "instruction")
    save_taxonomy_reference(valid_paths, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nProcessed data saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  - train_instruction.json")
    print("  - validation_instruction.json")
    print("  - test_instruction.json")
    print("  - sample_instruction.json")
    print("  - taxonomy_reference.json (for inference)")
    print("\nThe model will now be constrained to only output valid taxonomy paths!")


if __name__ == "__main__":
    main()
