import json
import re

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_all_leaf_nodes(taxonomy_dict, current_path=""):
    """Build all leaf nodes (final categories) from the taxonomy."""
    leaf_nodes = set()
    
    if isinstance(taxonomy_dict, dict):
        if not taxonomy_dict:  # Empty dict is a leaf
            if current_path:
                leaf_nodes.add(current_path)
        else:
            for key, value in taxonomy_dict.items():
                new_path = f"{current_path} > {key}" if current_path else key
                sub_leaves = build_all_leaf_nodes(value, new_path)
                if sub_leaves:
                    leaf_nodes.update(sub_leaves)
                elif not value or (isinstance(value, dict) and not value):
                    # This is a leaf node
                    leaf_nodes.add(new_path)
    
    elif isinstance(taxonomy_dict, list):
        # List items are leaf nodes
        for item in taxonomy_dict:
            new_path = f"{current_path} > {item}" if current_path else item
            leaf_nodes.add(new_path)
    
    return leaf_nodes

# Load files
corrected_paths = load_json('corrected_paths.json')
taxonomy_data = load_json('preprocessed_taxonomy.json')

# Build all leaf nodes
leaf_nodes = build_all_leaf_nodes(taxonomy_data['taxonomy'])

# Normalize paths (remove codes)
def normalize_path(path):
    return re.sub(r'\s*\(\d+\.\d+\)\s*', ' ', path).strip()

normalized_leaves = {normalize_path(leaf): leaf for leaf in leaf_nodes}

# Find errors - paths that don't exist
errors = []
for topic, data in corrected_paths.items():
    if 'your_taxonomy_path' in data:
        path = data['your_taxonomy_path']
        normalized = normalize_path(path)
        
        if normalized not in normalized_leaves:
            errors.append({
                'topic': topic,
                'invalid_path': path
            })

print(f"Found {len(errors)} paths that don't match taxonomy:\n")
for i, error in enumerate(errors[:10], 1):  # Show first 10
    print(f"{i}. {error['topic']}")
    print(f"   Path: {error['invalid_path']}")
    print()

# Find specific problematic terms
problematic_terms = {}
for error in errors:
    path = error['invalid_path']
    parts = [p.strip() for p in path.split('>')]
    last_part = parts[-1]
    # Remove code if present
    last_part_clean = re.sub(r'\s*\(\d+\.\d+\)\s*', '', last_part).strip()
    
    if last_part_clean not in problematic_terms:
        problematic_terms[last_part_clean] = []
    problematic_terms[last_part_clean].append(error['topic'])

print(f"\n\nMost common problematic terms:")
sorted_terms = sorted(problematic_terms.items(), key=lambda x: len(x[1]), reverse=True)
for term, topics in sorted_terms[:20]:
    print(f"\n'{term}' ({len(topics)} occurrences):")
    print(f"  Example topics: {', '.join(topics[:3])}")
