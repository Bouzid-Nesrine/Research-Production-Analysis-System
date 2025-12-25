import json
import re

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_all_valid_paths(taxonomy_dict, current_path="", all_paths=None):
    """Build all valid paths - both intermediate and leaf nodes."""
    if all_paths is None:
        all_paths = set()
    
    if isinstance(taxonomy_dict, dict):
        for key, value in taxonomy_dict.items():
            new_path = f"{current_path} > {key}" if current_path else key
            all_paths.add(new_path)  # Add this path as valid
            build_all_valid_paths(value, new_path, all_paths)
    
    elif isinstance(taxonomy_dict, list):
        # List items are also valid endpoints
        for item in taxonomy_dict:
            new_path = f"{current_path} > {item}" if current_path else item
            all_paths.add(new_path)
    
    return all_paths

# Load files
corrected_paths = load_json('corrected_paths.json')
taxonomy_data = load_json('preprocessed_taxonomy.json')

# Build all valid paths (intermediate and leaf)
all_valid_paths = build_all_valid_paths(taxonomy_data['taxonomy'])

# Normalize paths (remove codes and extra spaces)
def normalize_path(path):
    # Remove codes like (1.01)
    path = re.sub(r'\s*\(\d+\.\d+\)\s*', ' ', path)
    # Clean up spaces
    parts = [p.strip() for p in path.split('>')]
    return ' > '.join(parts)

normalized_valid = {normalize_path(p) for p in all_valid_paths}

# Check each path in corrected_paths
errors = []
for topic, data in corrected_paths.items():
    if 'your_taxonomy_path' in data:
        path = data['your_taxonomy_path']
        normalized = normalize_path(path)
        
        if normalized not in normalized_valid:
            errors.append({
                'topic': topic,
                'invalid_path': path,
                'normalized': normalized
            })

print(f"Found {len(errors)} truly invalid paths (not in taxonomy at all):\n")

if errors:
    for i, error in enumerate(errors[:20], 1):
        print(f"{i}. {error['topic']}")
        print(f"   Invalid: {error['invalid_path']}")
        
        # Try to find similar valid paths
        normalized = error['normalized']
        parts = [p.strip() for p in normalized.split('>')]
        
        # Find paths that start with the same prefix
        prefix = ' > '.join(parts[:-1]) if len(parts) > 1 else parts[0]
        similar = [p for p in normalized_valid if p.startswith(prefix)][:3]
        
        if similar:
            print(f"   Similar valid paths:")
            for s in similar:
                print(f"     - {s}")
        print()
else:
    print("All paths are valid!")

# Save errors
with open('missing_paths.json', 'w', encoding='utf-8') as f:
    json.dump(errors, f, indent=2, ensure_ascii=False)

print(f"\nTotal errors: {len(errors)}")
print(f"Errors saved to missing_paths.json")
