"""
Taxonomy Parser - Extract hierarchical paths from taxonomy JSON
"""

import json
from typing import List, Dict, Any
from pathlib import Path


class TaxonomyParser:
    """Parse and extract paths from hierarchical taxonomy"""
    
    def __init__(self, taxonomy_path: str):
        """
        Initialize parser with taxonomy file
        
        Args:
            taxonomy_path: Path to preprocessed_taxonomy.json
        """
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy = self._load_taxonomy()
        self.paths = []
        
    def _load_taxonomy(self) -> Dict:
        """Load taxonomy from JSON file"""
        with open(self.taxonomy_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('taxonomy', data)
    
    def extract_all_paths(self) -> List[Dict[str, Any]]:
        """
        Extract all hierarchical paths from taxonomy
        
        Returns:
            List of dictionaries containing path information
        """
        self.paths = []
        self._traverse_taxonomy(self.taxonomy, [])
        return self.paths
    
    def _traverse_taxonomy(self, node: Any, current_path: List[str]):
        """
        Recursively traverse taxonomy tree
        
        Args:
            node: Current node in taxonomy
            current_path: Path from root to current node
        """
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = current_path + [key]
                self._traverse_taxonomy(value, new_path)
        
        elif isinstance(node, list):
            # Leaf nodes - save the path
            for item in node:
                leaf_path = current_path + [item]
                self._save_path(leaf_path)
        
        elif node is None or (isinstance(node, dict) and len(node) == 0):
            # Empty node - save current path as leaf
            if current_path:
                self._save_path(current_path)
    
    def _save_path(self, path: List[str]):
        """
        Save a complete path with metadata
        
        Args:
            path: List of taxonomy levels from root to leaf
        """
        full_path = " > ".join(path)
        
        # Extract components
        domain = path[0] if len(path) > 0 else ""
        field = path[1] if len(path) > 1 else ""
        subfield = path[2] if len(path) > 2 else ""
        specialty = path[3] if len(path) > 3 else ""
        topic = path[4] if len(path) > 4 else ""
        
        # Create description for better semantic matching
        description = self._create_description(path)
        
        # Extract keywords
        keywords = self._extract_keywords(path)
        
        path_info = {
            "id": f"path_{len(self.paths):04d}",
            "full_path": full_path,
            "components": path,
            "level": len(path),
            "domain": domain,
            "field": field,
            "subfield": subfield,
            "specialty": specialty,
            "topic": topic,
            "description": description,
            "keywords": keywords,
        }
        
        self.paths.append(path_info)
    
    def _create_description(self, path: List[str]) -> str:
        """
        Create rich description for semantic matching
        
        Args:
            path: List of taxonomy levels
            
        Returns:
            Description string
        """
        if len(path) == 1:
            return f"Research in the domain of {path[0]}"
        
        elif len(path) == 2:
            return f"Research in {path[1]} within {path[0]}"
        
        elif len(path) == 3:
            return f"Research in {path[2]}, a subfield of {path[1]} in {path[0]}"
        
        elif len(path) == 4:
            return (
                f"Research focusing on {path[3]} within {path[2]}, "
                f"which is part of {path[1]} in the {path[0]} domain"
            )
        
        else:  # len >= 5
            return (
                f"Specialized research on {path[-1]} in the area of {path[-2]}, "
                f"within {path[-3]}, part of {path[1]} in {path[0]}"
            )
    
    def _extract_keywords(self, path: List[str]) -> List[str]:
        """
        Extract keywords from path components
        
        Args:
            path: List of taxonomy levels
            
        Returns:
            List of keywords
        """
        keywords = []
        
        for component in path:
            # Split on common delimiters
            words = component.replace('-', ' ').replace('_', ' ').split()
            keywords.extend(words)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def get_paths_by_level(self, level: int) -> List[Dict[str, Any]]:
        """
        Get all paths at a specific level
        
        Args:
            level: Taxonomy level (1-5)
            
        Returns:
            List of path dictionaries at specified level
        """
        return [p for p in self.paths if p['level'] == level]
    
    def get_paths_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get all paths within a specific domain
        
        Args:
            domain: Domain name
            
        Returns:
            List of path dictionaries in specified domain
        """
        return [p for p in self.paths if p['domain'] == domain]
    
    def save_paths(self, output_path: str):
        """
        Save extracted paths to JSON file
        
        Args:
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.paths, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.paths)} paths to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about taxonomy paths
        
        Returns:
            Dictionary with statistics
        """
        from collections import Counter
        
        level_counts = Counter(p['level'] for p in self.paths)
        domain_counts = Counter(p['domain'] for p in self.paths)
        
        return {
            "total_paths": len(self.paths),
            "paths_by_level": dict(level_counts),
            "paths_by_domain": dict(domain_counts),
            "max_level": max(level_counts.keys()) if level_counts else 0,
            "domains": list(domain_counts.keys()),
        }


def main():
    """Example usage"""
    from config import TAXONOMY_PATH
    
    # Initialize parser
    parser = TaxonomyParser(TAXONOMY_PATH)
    
    # Extract all paths
    paths = parser.extract_all_paths()
    
    # Print statistics
    stats = parser.get_statistics()
    print("\n=== Taxonomy Statistics ===")
    print(f"Total paths: {stats['total_paths']}")
    print(f"Max level: {stats['max_level']}")
    print(f"\nPaths by level:")
    for level, count in sorted(stats['paths_by_level'].items()):
        print(f"  Level {level}: {count} paths")
    print(f"\nDomains: {len(stats['domains'])}")
    for domain in sorted(stats['domains']):
        print(f"  - {domain}")
    
    # Show example paths
    print("\n=== Example Paths ===")
    for i, path in enumerate(paths[:5], 1):
        print(f"\n{i}. {path['full_path']}")
        print(f"   Level: {path['level']}")
        print(f"   Description: {path['description']}")
        print(f"   Keywords: {', '.join(path['keywords'][:5])}...")
    
    # Save paths
    output_path = Path(__file__).parent / "taxonomy_paths.json"
    parser.save_paths(output_path)


if __name__ == "__main__":
    main()
