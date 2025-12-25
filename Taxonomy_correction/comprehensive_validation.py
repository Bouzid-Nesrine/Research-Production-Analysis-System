import json

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def comprehensive_validation(mapping_file, taxonomy_file):
    """
    Perform comprehensive validation including:
    1. Path structure correctness
    2. Verify each level in the path exists
    3. Check for formatting issues
    """
    print("Loading files...")
    mapping = load_json(mapping_file)
    taxonomy_data = load_json(taxonomy_file)
    taxonomy_root = taxonomy_data.get('taxonomy', taxonomy_data)
    
    print(f"\nTotal topics in mapping: {len(mapping)}")
    
    # Validation checks
    issues = {
        'empty_paths': [],
        'malformed_paths': [],
        'path_traversal_errors': [],
        'formatting_issues': []
    }
    
    def verify_path_exists(path):
        """Verify each level of the path exists in taxonomy."""
        parts = [p.strip() for p in path.split(' > ')]
        current = taxonomy_root
        
        for i, part in enumerate(parts):
            if isinstance(current, dict):
                if part in current:
                    current = current[part]
                else:
                    return False, f"Level {i+1} '{part}' not found in taxonomy", parts[:i+1]
            elif isinstance(current, list):
                if part in current:
                    return True, "Valid", parts  # Reached leaf
                else:
                    return False, f"Leaf node '{part}' not in list", parts[:i]
            else:
                return False, f"Unexpected structure at level {i}", parts[:i]
        
        return True, "Valid", parts
    
    print("\nPerforming comprehensive validation...")
    valid_count = 0
    
    for topic_name, topic_data in mapping.items():
        path = topic_data.get('your_taxonomy_path', '')
        
        # Check 1: Empty path
        if not path:
            issues['empty_paths'].append({
                'topic': topic_name,
                'path': path
            })
            continue
        
        # Check 2: Malformed path (doesn't contain ' > ')
        if ' > ' not in path:
            issues['malformed_paths'].append({
                'topic': topic_name,
                'path': path,
                'reason': 'Missing separator " > "'
            })
            continue
        
        # Check 3: Formatting issues (extra spaces, etc.)
        path_parts = path.split(' > ')
        has_formatting_issue = False
        for part in path_parts:
            if part != part.strip():
                has_formatting_issue = True
                break
            if '  ' in part:  # Double spaces
                has_formatting_issue = True
                break
        
        if has_formatting_issue:
            issues['formatting_issues'].append({
                'topic': topic_name,
                'path': path,
                'reason': 'Extra spaces or formatting issues'
            })
        
        # Check 4: Path traversal
        is_valid, message, validated_parts = verify_path_exists(path)
        if not is_valid:
            issues['path_traversal_errors'].append({
                'topic': topic_name,
                'path': path,
                'error': message,
                'validated_up_to': ' > '.join(validated_parts)
            })
        else:
            valid_count += 1
    
    # Print results
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION RESULTS")
    print("="*80)
    print(f"\nTotal topics: {len(mapping)}")
    print(f"✓ Valid paths: {valid_count}")
    
    total_issues = sum(len(v) for v in issues.values())
    print(f"✗ Total issues found: {total_issues}")
    
    for issue_type, issue_list in issues.items():
        if issue_list:
            print(f"\n{issue_type.replace('_', ' ').title()}: {len(issue_list)}")
            for i, issue in enumerate(issue_list[:5], 1):
                print(f"  {i}. {issue['topic']}")
                print(f"     Path: {issue['path']}")
                if 'reason' in issue:
                    print(f"     Reason: {issue['reason']}")
                if 'error' in issue:
                    print(f"     Error: {issue['error']}")
                if 'validated_up_to' in issue:
                    print(f"     Valid up to: {issue['validated_up_to']}")
            
            if len(issue_list) > 5:
                print(f"  ... and {len(issue_list) - 5} more")
    
    # Save detailed report
    if total_issues > 0:
        report_file = 'comprehensive_validation_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(issues, f, indent=2, ensure_ascii=False)
        print(f"\n\nDetailed report saved to: {report_file}")
    else:
        print("\n" + "="*80)
        print("✓✓✓ ALL VALIDATIONS PASSED! ✓✓✓")
        print("="*80)
        print("\n✓ All paths are correctly formatted")
        print("✓ All paths exist in the taxonomy")
        print("✓ All path levels are valid")
        print("✓ No formatting issues detected")
    
    return issues

if __name__ == "__main__":
    mapping_file = "./mapping_corrected__preprocess.json"
    taxonomy_file = "../Taxonomy Building/preprocessed_taxonomy.json"
    
    issues = comprehensive_validation(mapping_file, taxonomy_file)
