#!/usr/bin/env python3
"""
Validate that training/test data classification paths match the taxonomy leaf paths.
Run this before training to ensure data compatibility.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set

def extract_leaf_paths(taxonomy_dict: Dict, prefix: str = "") -> List[str]:
    """Extract only leaf paths (complete paths to terminal nodes)."""
    leaf_paths = []
    
    if isinstance(taxonomy_dict, dict):
        for key, value in taxonomy_dict.items():
            current_path = f"{prefix} > {key}" if prefix else key
            
            if isinstance(value, dict) and value:  # Has children
                leaf_paths.extend(extract_leaf_paths(value, current_path))
            elif isinstance(value, list) and value:  # List of leaf nodes
                for item in value:
                    leaf_path = f"{current_path} > {item}"
                    leaf_paths.append(leaf_path)
            elif not value or (isinstance(value, dict) and not value):  # Terminal node
                leaf_paths.append(current_path)
    
    return leaf_paths

def load_json_files(data_dir: str) -> List[Dict]:
    """Load all JSON files from directory."""
    all_articles = []
    json_files = list(Path(data_dir).glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    articles = data
                elif isinstance(data, dict):
                    articles = data.get('articles', [data])
                else:
                    continue
                
                valid_articles = [a for a in articles if isinstance(a, dict)]
                all_articles.extend(valid_articles)
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
            continue
    
    return all_articles

def main():
    # Paths
    TAXONOMY_PATH = "../Taxonomy Building/preprocessed_taxonomy.json"
    TRAIN_DATA_DIR = "./train_data"
    TEST_DATA_DIR = "./test_data"
    
    print("="*70)
    print("TAXONOMY & DATA VALIDATION")
    print("="*70)
    
    # Load taxonomy
    print("\n1. Loading taxonomy...")
    with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
        taxonomy_data = json.load(f)
    
    taxonomy = taxonomy_data.get('taxonomy', taxonomy_data)
    leaf_paths = sorted(list(set(extract_leaf_paths(taxonomy))))
    
    print(f"   ✅ Taxonomy loaded: {len(leaf_paths)} leaf paths")
    
    # Analyze taxonomy depth
    depths = [len(path.split(' > ')) for path in leaf_paths]
    depth_counts = Counter(depths)
    print(f"   📊 Depth distribution:")
    for depth in sorted(depth_counts.keys()):
        print(f"      Level {depth}: {depth_counts[depth]} paths")
    
    # Load training data
    print("\n2. Loading training data...")
    train_articles = load_json_files(TRAIN_DATA_DIR)
    print(f"   ✅ Loaded {len(train_articles)} training articles")
    
    # Load test data
    print("\n3. Loading test data...")
    test_articles = load_json_files(TEST_DATA_DIR)
    print(f"   ✅ Loaded {len(test_articles)} test articles")
    
    # Extract classification paths from data
    print("\n4. Validating classification paths...")
    
    taxonomy_set = set(leaf_paths)
    train_paths = []
    test_paths = []
    
    train_missing = 0
    test_missing = 0
    
    for article in train_articles:
        path = article.get('classification_path', '').strip()
        if path:
            train_paths.append(path)
            if path not in taxonomy_set:
                train_missing += 1
    
    for article in test_articles:
        path = article.get('classification_path', '').strip()
        if path:
            test_paths.append(path)
            if path not in taxonomy_set:
                test_missing += 1
    
    # Results
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print("="*70)
    
    # Training data
    unique_train = len(set(train_paths))
    coverage_train = (unique_train / len(leaf_paths)) * 100
    print(f"\n📊 Training Data:")
    print(f"   • Total articles with classification: {len(train_paths)}")
    print(f"   • Unique classification paths: {unique_train}")
    print(f"   • Taxonomy coverage: {coverage_train:.1f}% ({unique_train}/{len(leaf_paths)})")
    print(f"   • Paths NOT in taxonomy: {train_missing}")
    
    if train_missing > 0:
        print(f"   ⚠️  WARNING: {train_missing} training articles have invalid paths!")
    else:
        print(f"   ✅ All training paths match taxonomy!")
    
    # Test data
    unique_test = len(set(test_paths))
    coverage_test = (unique_test / len(leaf_paths)) * 100
    print(f"\n📊 Test Data:")
    print(f"   • Total articles with classification: {len(test_paths)}")
    print(f"   • Unique classification paths: {unique_test}")
    print(f"   • Taxonomy coverage: {coverage_test:.1f}% ({unique_test}/{len(leaf_paths)})")
    print(f"   • Paths NOT in taxonomy: {test_missing}")
    
    if test_missing > 0:
        print(f"   ⚠️  WARNING: {test_missing} test articles have invalid paths!")
    else:
        print(f"   ✅ All test paths match taxonomy!")
    
    # Show most common paths
    print(f"\n📈 Top 10 most common training paths:")
    path_counts = Counter(train_paths)
    for i, (path, count) in enumerate(path_counts.most_common(10), 1):
        print(f"   {i:2d}. [{count:4d}x] {path[:80]}...")
    
    # Find paths not in data
    unused_paths = taxonomy_set - set(train_paths) - set(test_paths)
    print(f"\n📉 Taxonomy paths with NO data: {len(unused_paths)}")
    if len(unused_paths) <= 20:
        for path in sorted(list(unused_paths))[:20]:
            print(f"   • {path}")
    else:
        print(f"   (Showing first 20 of {len(unused_paths)})")
        for path in sorted(list(unused_paths))[:20]:
            print(f"   • {path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    
    if train_missing == 0 and test_missing == 0:
        print("✅ All data paths are valid and match the taxonomy!")
        print("✅ Ready to train the model!")
    else:
        print(f"⚠️  WARNING: Found {train_missing + test_missing} invalid paths")
        print("   Please clean the data before training")
    
    print(f"\n💡 The model will classify into {len(leaf_paths)} leaf categories")
    print(f"💡 Training data covers {coverage_train:.1f}% of taxonomy")
    print(f"💡 Test data covers {coverage_test:.1f}% of taxonomy")

if __name__ == "__main__":
    main()
