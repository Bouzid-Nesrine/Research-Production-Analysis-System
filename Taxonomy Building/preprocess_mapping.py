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

def normalize_path_component(component):
    """Normalize a single component of the path."""
    component = component.strip()
    component = remove_numeric_codes(component)
    component = expand_abbreviations(component)
    # Remove extra whitespace
    component = ' '.join(component.split())
    return component

def normalize_taxonomy_path(path):
    """
    Normalize taxonomy path by:
    - Removing numeric codes like (1.02)
    - Expanding abbreviations
    - Maintaining > delimiter structure
    """
    if not path or not isinstance(path, str):
        return path
    
    # Split by > delimiter
    components = path.split('>')
    
    # Normalize each component
    normalized_components = [normalize_path_component(comp) for comp in components]
    
    # Rejoin with > delimiter (with spaces around it for consistency)
    return ' > '.join(normalized_components)

def preprocess_mapping_file(input_file, output_file):
    """
    Preprocess mapping file to normalize only the 'your_taxonomy_path' values.
    """
    print("Loading mapping file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_count = 0
    total_count = len(data)
    
    print(f"Processing {total_count} entries...")
    
    # Process each entry
    for key, value in data.items():
        if isinstance(value, dict) and 'your_taxonomy_path' in value:
            original_path = value['your_taxonomy_path']
            normalized_path = normalize_taxonomy_path(original_path)
            
            if original_path != normalized_path:
                value['your_taxonomy_path'] = normalized_path
                processed_count += 1
    
    # Save processed mapping
    print(f"Saving preprocessed mapping to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing complete!")
    print("=" * 60)
    print(f"✓ Input file: {input_file}")
    print(f"✓ Output file: {output_file}")
    print(f"✓ Total entries: {total_count}")
    print(f"✓ Modified paths: {processed_count}")
    print(f"✓ Unchanged paths: {total_count - processed_count}")
    print("=" * 60)
    print("\nTransformations applied:")
    print("  • Removed numeric codes like (1.02), (3.01), etc.")
    print("  • Expanded abbreviations (NLP → Natural Language Processing)")
    print("  • Normalized whitespace and formatting")
    print("  • Maintained > delimiter structure")
    print("=" * 60)

def show_sample_transformations(input_file, num_samples=5):
    """Show sample transformations for verification."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\nSample transformations:")
    print("=" * 60)
    
    count = 0
    for key, value in data.items():
        if isinstance(value, dict) and 'your_taxonomy_path' in value:
            original_path = value['your_taxonomy_path']
            normalized_path = normalize_taxonomy_path(original_path)
            
            if original_path != normalized_path and count < num_samples:
                print(f"\n{count + 1}. Entry: {key}")
                print(f"   Before: {original_path}")
                print(f"   After:  {normalized_path}")
                count += 1
                
                if count >= num_samples:
                    break
    
    if count == 0:
        print("\nNo transformations needed - all paths are already normalized!")
    
    print("=" * 60)

if __name__ == "__main__":
    input_file = "mapping_final_version.json"
    output_file = "mapping_preprocessed.json"
    
    print("=" * 60)
    print("MAPPING FILE PREPROCESSING")
    print("=" * 60)
    
    # Show sample transformations first
    show_sample_transformations(input_file, num_samples=5)
    
    # Run preprocessing
    preprocess_mapping_file(input_file, output_file)
    
    print("\n✓ All done! 🎉")
