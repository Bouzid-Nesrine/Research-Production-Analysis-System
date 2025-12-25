import json
from typing import Set

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_all_leaf_nodes(taxonomy_dict):
    """
    Extract all leaf nodes (final categories) from the taxonomy.
    A leaf is either a string in a list or a key with an empty list/no children.
    """
    leaf_nodes = set()
    
    def traverse(node, path_parts):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = path_parts + [key]
                if not value or (isinstance(value, list) and len(value) == 0):
                    # This is a leaf node
                    leaf_nodes.add(key)
                else:
                    traverse(value, new_path)
        elif isinstance(node, list):
            for item in node:
                # Items in lists are leaf nodes
                leaf_nodes.add(item)
    
    traverse(taxonomy_dict, [])
    return leaf_nodes

def validate_leaf_nodes(mapping_file, taxonomy_file):
    """
    Validate that all leaf nodes (final categories) in mapping paths exist in taxonomy.
    """
    print("Loading files...")
    mapping = load_json(mapping_file)
    taxonomy_data = load_json(taxonomy_file)
    
    print("Extracting all leaf nodes from taxonomy...")
    taxonomy_root = taxonomy_data.get('taxonomy', taxonomy_data)
    valid_leaves = extract_all_leaf_nodes(taxonomy_root)
    
    print(f"Found {len(valid_leaves)} valid leaf nodes in taxonomy")
    
    print("\nValidating leaf nodes in mapping paths...")
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
        
        # Extract the leaf node (last part of the path)
        path_parts = path.split(' > ')
        leaf_node = path_parts[-1].strip() if path_parts else ''
        
        # Check if leaf node exists in valid leaves
        if leaf_node not in valid_leaves:
            errors.append({
                'topic': topic_name,
                'error': 'Leaf node not found in taxonomy',
                'leaf': leaf_node,
                'full_path': path
            })
        else:
            valid_count += 1
    
    print(f"\nLeaf Node Validation Results:")
    print(f"Total topics: {len(mapping)}")
    print(f"Valid leaf nodes: {valid_count}")
    print(f"Invalid leaf nodes: {len(errors)}")
    
    if errors:
        print("\n" + "="*80)
        print("LEAF NODE ERRORS FOUND:")
        print("="*80)
        for i, error in enumerate(errors[:50], 1):  # Show first 50
            print(f"\n{i}. Topic: {error['topic']}")
            print(f"   Error: {error['error']}")
            print(f"   Invalid Leaf: {error['leaf']}")
            print(f"   Full Path: {error['full_path']}")
            
            # Try to find similar leaf nodes
            leaf = error['leaf']
            similar = [v for v in valid_leaves if leaf.lower() in v.lower() or v.lower() in leaf.lower()]
            if similar:
                print(f"   Similar valid leaves:")
                for s in similar[:3]:
                    print(f"     - {s}")
        
        if len(errors) > 50:
            print(f"\n... and {len(errors) - 50} more errors")
    
        # Save errors to file
        error_file = 'leaf_node_validation_errors.json'
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"\n\nErrors saved to: {error_file}")
    else:
        print("\n✓ All leaf nodes are valid!")
    
    # Print some sample valid leaves for reference
    print("\n" + "="*80)
    print("Sample valid leaf nodes (first 20):")
    print("="*80)
    for i, leaf in enumerate(sorted(valid_leaves)[:20], 1):
        print(f"{i}. {leaf}")
    
    return errors

if __name__ == "__main__":
    mapping_file = "./mapping_corrected__preprocess.json"
    taxonomy_file = "../Taxonomy Building/preprocessed_taxonomy.json"
    
    errors = validate_leaf_nodes(mapping_file, taxonomy_file)
