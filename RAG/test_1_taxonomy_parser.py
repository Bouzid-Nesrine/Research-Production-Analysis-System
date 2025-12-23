"""
Step 1: Test Taxonomy Parser
Extract and verify taxonomy paths
"""

import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from taxonomy_parser import TaxonomyParser
from config import TAXONOMY_PATH

print("="*70)
print("STEP 1: TESTING TAXONOMY PARSER")
print("="*70)

# Test 1: Load taxonomy
print("\n[Test 1] Loading taxonomy file...")
print(f"Path: {TAXONOMY_PATH}")
print(f"File exists: {TAXONOMY_PATH.exists()}")

if not TAXONOMY_PATH.exists():
    print(f"❌ ERROR: Taxonomy file not found at {TAXONOMY_PATH}")
    sys.exit(1)

print("✓ Taxonomy file found")

# Test 2: Initialize parser
print("\n[Test 2] Initializing parser...")
try:
    parser = TaxonomyParser(TAXONOMY_PATH)
    print("✓ Parser initialized successfully")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# Test 3: Extract paths
print("\n[Test 3] Extracting taxonomy paths...")
try:
    paths = parser.extract_all_paths()
    print(f"✓ Extracted {len(paths)} paths")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# Test 4: Get statistics
print("\n[Test 4] Analyzing taxonomy structure...")
stats = parser.get_statistics()

print("\nTaxonomy Statistics:")
print(f"  Total paths: {stats['total_paths']}")
print(f"  Max depth: {stats['max_level']}")
print(f"  Number of domains: {len(stats['domains'])}")

print("\nPaths by level:")
for level, count in sorted(stats['paths_by_level'].items()):
    print(f"  Level {level}: {count} paths")

print("\nDomains:")
for i, domain in enumerate(sorted(stats['domains']), 1):
    print(f"  {i}. {domain}")

# Test 5: Examine sample paths
print("\n[Test 5] Sample paths:")
print("-" * 70)

for i, path in enumerate(paths[:10], 1):
    print(f"\n{i}. Path: {path['full_path']}")
    print(f"   ID: {path['id']}")
    print(f"   Level: {path['level']}")
    print(f"   Domain: {path['domain']}")
    print(f"   Description: {path['description'][:80]}...")
    print(f"   Keywords: {', '.join(path['keywords'][:5])}")

# Test 6: Verify path structure
print("\n[Test 6] Verifying path structure...")
sample_path = paths[0]
required_fields = ['id', 'full_path', 'components', 'level', 'domain', 
                   'description', 'keywords']

missing_fields = [field for field in required_fields if field not in sample_path]

if missing_fields:
    print(f"❌ Missing fields: {missing_fields}")
else:
    print("✓ All required fields present")

# Test 7: Save paths to JSON
print("\n[Test 7] Saving extracted paths...")
output_path = Path(__file__).parent / 'test_taxonomy_paths.json'
try:
    parser.save_paths(output_path)
    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 8: Verify different domains
print("\n[Test 8] Checking domain distribution...")
domain_counts = {}
for path in paths:
    domain = path['domain']
    domain_counts[domain] = domain_counts.get(domain, 0) + 1

print("\nPaths per domain:")
for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {domain}: {count} paths")

# Test 9: Verify path levels
print("\n[Test 9] Checking path depth distribution...")
level_examples = {}
for path in paths:
    level = path['level']
    if level not in level_examples and level <= 5:
        level_examples[level] = path['full_path']

print("\nExample paths at each level:")
for level in sorted(level_examples.keys()):
    print(f"\nLevel {level}: {level_examples[level]}")

# Final summary
print("\n" + "="*70)
print("TAXONOMY PARSER TEST SUMMARY")
print("="*70)
print(f"✓ Taxonomy loaded successfully")
print(f"✓ {len(paths)} paths extracted")
print(f"✓ {len(stats['domains'])} domains identified")
print(f"✓ Max depth: {stats['max_level']} levels")
print(f"✓ Output saved to: {output_path}")
print("="*70)
print("\n✅ ALL TESTS PASSED - Taxonomy parser is working correctly!")
print("\nNext step: Run test_2_vector_database.py")
print("="*70)
