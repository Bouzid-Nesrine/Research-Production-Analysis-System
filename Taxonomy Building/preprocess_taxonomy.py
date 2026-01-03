import json
import re

def expand_abbreviations(text):
    """Expand common abbreviations to full forms."""
    abbreviations = {
        'NLP': 'Natural Language Processing',
        'AI': 'Artificial Intelligence',
        'ML': 'Machine Learning',
        'DL': 'Deep Learning',
        'CV': 'Computer Vision',
        'RNA': 'Ribonucleic Acid',
        'DNA': 'Deoxyribonucleic Acid',
        'GPU': 'Graphics Processing Unit',
        'CPU': 'Central Processing Unit',
        'API': 'Application Programming Interface',
        'UI': 'User Interface',
        'UX': 'User Experience',
        'IoT': 'Internet of Things',
        'VR': 'Virtual Reality',
        'AR': 'Augmented Reality',
        'MR': 'Mixed Reality',
        'CAD': 'Computer-Aided Design',
        'CAM': 'Computer-Aided Manufacturing',
        'CNC': 'Computer Numerical Control',
        'HVAC': 'Heating Ventilation and Air Conditioning',
        'MRI': 'Magnetic Resonance Imaging',
        'CT': 'Computed Tomography',
        'OSI': 'Open Systems Interconnection',
        'TCP': 'Transmission Control Protocol',
        'IP': 'Internet Protocol',
        'VPN': 'Virtual Private Network',
        'SQL': 'Structured Query Language',
        'NoSQL': 'Not Only Structured Query Language',
        'ACID': 'Atomicity Consistency Isolation Durability',
        'ETL': 'Extract Transform Load',
        'OLAP': 'Online Analytical Processing',
        'GIS': 'Geographic Information System',
        'STEM': 'Science Technology Engineering and Mathematics',
        'CRISPR': 'Clustered Regularly Interspaced Short Palindromic Repeats',
        '3D': 'Three-Dimensional',
        '2D': 'Two-Dimensional',
        'VLSI': 'Very-Large-Scale Integration',
        'ACM': 'Association for Computing Machinery',
        'CCS': 'Computing Classification System',
        'UNESCO': 'United Nations Educational Scientific and Cultural Organization',
        'FOS': 'Fields of Science',
        'OECD': 'Organisation for Economic Co-operation and Development',
        'AGI': 'Artificial General Intelligence'
    }
    
    # Replace standalone abbreviations (word boundaries)
    for abbr, full in abbreviations.items():
        # Match abbreviation as whole word or with 's' suffix
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
    
    return text

def remove_numeric_codes(text):
    """Remove numeric codes like (1.02) from text."""
    # Remove patterns like (1.02), (1.01), etc.
    text = re.sub(r'\s*\(\d+\.\d+\)', '', text)
    return text.strip()

def normalize_label(text):
    """Normalize labels by expanding abbreviations and cleaning."""
    text = remove_numeric_codes(text)
    text = expand_abbreviations(text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def process_taxonomy_recursive(data, path=""):
    """Recursively process taxonomy structure."""
    if isinstance(data, dict):
        processed = {}
        for key, value in data.items():
            # Normalize the key
            normalized_key = normalize_label(key)
            current_path = f"{path} > {normalized_key}" if path else normalized_key
            
            # Process the value recursively
            processed[normalized_key] = process_taxonomy_recursive(value, current_path)
        return processed
    
    elif isinstance(data, list):
        # Process list items
        return [normalize_label(item) for item in data]
    
    else:
        # Base case: return normalized string
        return normalize_label(str(data))

def preprocess_taxonomy(input_file, output_file):
    """
    Preprocess taxonomy file:
    - Remove numeric codes like (1.02)
    - Expand abbreviations to full forms
    - Ensure consistent label formulation
    - Maintain hierarchy structure
    - Use consistent delimiter (>)
    """
    # Load input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process metadata separately (optional)
    processed_data = {}
    
    if 'metadata' in data:
        # Update metadata
        metadata = data['metadata'].copy()
        if 'description' in metadata:
            metadata['description'] = normalize_label(metadata['description'])
        processed_data['metadata'] = metadata
    
    # Process taxonomy
    if 'taxonomy' in data:
        processed_data['taxonomy'] = process_taxonomy_recursive(data['taxonomy'])
    else:
        # If no taxonomy key, process entire structure
        processed_data = process_taxonomy_recursive(data)
    
    # Save processed taxonomy
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Preprocessing complete!")
    print(f"✓ Input: {input_file}")
    print(f"✓ Output: {output_file}")
    print(f"✓ Numeric codes removed")
    print(f"✓ Abbreviations expanded")
    print(f"✓ Labels normalized")

def generate_hierarchy_paths(input_file, output_file):
    """
    Generate a list of all hierarchy paths using > delimiter.
    Example: Computer Science > Machine Learning > Natural Language Processing
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    paths = []
    
    def extract_paths(data, current_path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path} > {key}" if current_path else key
                paths.append(new_path)
                extract_paths(value, new_path)
        elif isinstance(data, list):
            for item in data:
                leaf_path = f"{current_path} > {item}" if current_path else item
                paths.append(leaf_path)
    
    # Extract paths from taxonomy
    if 'taxonomy' in data:
        extract_paths(data['taxonomy'])
    else:
        extract_paths(data)
    
    # Save paths
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in paths:
            f.write(path + '\n')
    
    print(f"✓ Generated {len(paths)} hierarchy paths")
    print(f"✓ Saved to: {output_file}")

if __name__ == "__main__":
    # File paths
    input_file = "final_combined_taxonomy.json"
    output_file = "preprocessed_taxonomy.json"
    paths_file = "taxonomy_paths.txt"
    
    # Run preprocessing
    print("Starting taxonomy preprocessing...")
    print("=" * 50)
    
    preprocess_taxonomy(input_file, output_file)
    
    print("\n" + "=" * 50)
    print("Generating hierarchy paths...")
    print("=" * 50)
    
    generate_hierarchy_paths(output_file, paths_file)
    
    print("\n" + "=" * 50)
    print("All done! ")
    print("=" * 50)
