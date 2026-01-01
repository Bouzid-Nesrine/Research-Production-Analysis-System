import json
from typing import Dict, List, Set, Tuple

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_all_paths_from_taxonomy(taxonomy_dict, current_path=""):
    """
    Recursively extract all valid paths from the taxonomy.
    Returns a set of all valid paths.
    """
    all_paths = set()
    
    def traverse(node, path_parts):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = path_parts + [key]
                # Add the current path
                all_paths.add(" > ".join(new_path))
                # Recurse
                traverse(value, new_path)
        elif isinstance(node, list):
            for item in node:
                new_path = path_parts + [item]
                # Add the leaf path
                all_paths.add(" > ".join(new_path))
    
    traverse(taxonomy_dict, [])
    return all_paths

def validate_mapping_paths(mapping_file, taxonomy_file):
    """
    Validate all paths in mapping file against taxonomy.
    Returns errors found.
    """
    print("Loading files...")
    mapping = load_json(mapping_file)
    taxonomy_data = load_json(taxonomy_file)
    
    print("Extracting all valid paths from taxonomy...")
    # Get the taxonomy root
    taxonomy_root = taxonomy_data.get('taxonomy', taxonomy_data)
    valid_paths = extract_all_paths_from_taxonomy(taxonomy_root)
    
    print(f"Found {len(valid_paths)} valid paths in taxonomy")
    
    print("\nValidating mapping paths...")
    errors = []
    valid_count = 0
    
    for topic_name, topic_data in mapping.items():
        path = topic_data.get('your_taxonomy_path', '')
        
        if not path:
            errors.append({
                'topic': topic_name,
                'error': 'Empty path',
                'path': path
            })
            continue
        
        # Check if path exists in valid paths
        if path not in valid_paths:
            errors.append({
                'topic': topic_name,
                'error': 'Path not found in taxonomy',
                'path': path
            })
        else:
            valid_count += 1
    
    print(f"\nValidation Results:")
    print(f"Total topics: {len(mapping)}")
    print(f"Valid paths: {valid_count}")
    print(f"Invalid paths: {len(errors)}")
    
    if errors:
        print("\n" + "="*80)
        print("ERRORS FOUND:")
        print("="*80)
        for i, error in enumerate(errors, 1):
            print(f"\n{i}. Topic: {error['topic']}")
            print(f"   Error: {error['error']}")
            print(f"   Path: {error['path']}")
            
            # Try to suggest closest matches
            if error['error'] == 'Path not found in taxonomy':
                path_parts = error['path'].split(' > ')
                leaf = path_parts[-1] if path_parts else ''
                
                # Find paths containing the leaf
                matching_paths = [p for p in valid_paths if leaf in p]
                if matching_paths:
                    print(f"   Possible correct paths containing '{leaf}':")
                    for mp in matching_paths[:5]:  # Show max 5 suggestions
                        print(f"     - {mp}")
    
    # Save errors to file
    if errors:
        error_file = 'path_validation_errors.json'
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"\n\nErrors saved to: {error_file}")
    else:
        print("\n✓ All paths are valid!")
    
    return errors

if __name__ == "__main__":
    mapping_file = "./mapping_corrected__preprocess.json"
    taxonomy_file = "../Taxonomy Building/preprocessed_taxonomy.json"
    
    errors = validate_mapping_paths(mapping_file, taxonomy_file)

# Print errors
print(f"Found {len(errors)} invalid paths:\n")
for error in errors:
    print(f"Topic: {error['topic']}")
    print(f"  Invalid path: {error['invalid_path']}")
    print(f"  Normalized: {error['normalized']}")
    print()

# Save errors to file
with open('validation_errors.json', 'w', encoding='utf-8') as f:
    json.dump(errors, f, indent=2, ensure_ascii=False)

print(f"\nErrors saved to validation_errors.json")
