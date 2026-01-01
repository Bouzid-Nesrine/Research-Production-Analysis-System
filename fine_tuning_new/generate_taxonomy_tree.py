import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

def load_taxonomy(file_path):
    """Load taxonomy from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['taxonomy']

def build_graph(taxonomy, parent=None, graph=None, level=0, max_level=3):
    """
    Build a networkx graph from taxonomy structure.
    max_level: Maximum depth to display (None for all levels)
    """
    if graph is None:
        graph = nx.DiGraph()
    
    if level > max_level and max_level is not None:
        return graph
    
    for key, value in taxonomy.items():
        node_name = key
        graph.add_node(node_name, level=level)
        
        if parent is not None:
            graph.add_edge(parent, node_name)
        
        if isinstance(value, dict):
            build_graph(value, node_name, graph, level + 1, max_level)
        elif isinstance(value, list):
            # For lists, we can optionally add items as leaf nodes
            # Uncomment below if you want to show list items
            # for item in value[:3]:  # Limit to first 3 items to avoid clutter
            #     item_node = f"{node_name}:{item}"
            #     graph.add_node(item_node, level=level + 1)
            #     graph.add_edge(node_name, item_node)
            pass
    
    return graph

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Create a hierarchical layout where nodes at the same level are in the same row.
    """
    if root is None:
        # Find root nodes (nodes with no predecessors)
        roots = [n for n, d in G.in_degree() if d == 0]
        if len(roots) == 1:
            root = roots[0]
        else:
            # Create a dummy root
            root = "ROOT"
            G.add_node(root, level=-1)
            for r in roots:
                G.add_edge(root, r)
    
    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                    vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent=root, parsed=parsed)
        return pos
    
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def visualize_taxonomy_hierarchical(taxonomy, output_file='taxonomy_tree.png', max_level=3):
    """
    Create a hierarchical tree visualization of the taxonomy.
    """
    # Build the graph
    G = build_graph(taxonomy, max_level=max_level)
    
    # Create figure with larger size
    plt.figure(figsize=(28, 18))
    
    # Use custom hierarchical layout
    print("Creating hierarchical layout...")
    pos = hierarchy_pos(G, width=2.0, vert_gap=0.15)
    
    # Remove dummy root if it was added
    if "ROOT" in G.nodes():
        G.remove_node("ROOT")
        del pos["ROOT"]
    
    # Get node levels for coloring
    levels = nx.get_node_attributes(G, 'level')
    colors = [levels[node] for node in G.nodes()]
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=colors,
            node_size=2000,
            cmap=plt.cm.Set3,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            alpha=0.9,
            arrowsize=12,
            linewidths=2,
            arrowstyle='->',
            node_shape='o')
    
    plt.title(f'Research Taxonomy Tree (Levels 0-{max_level})', 
              fontsize=22, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Tree visualization saved to {output_file}")
    plt.close()

def visualize_taxonomy_radial(taxonomy, output_file='taxonomy_tree_radial.png', max_level=2):
    """
    Create a radial tree visualization of the taxonomy.
    """
    # Build the graph
    G = build_graph(taxonomy, max_level=max_level)
    
    # Create figure
    plt.figure(figsize=(20, 20))
    
    # Use circular layout
    pos = nx.spring_layout(G, k=2, iterations=50, scale=2)
    
    # Get node levels for coloring
    levels = nx.get_node_attributes(G, 'level')
    colors = [levels[node] for node in G.nodes()]
    
    # Draw the graph
    nx.draw(G, pos,
            node_color=colors,
            node_size=2000,
            cmap=plt.cm.viridis,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            arrows=False,
            edge_color='lightgray',
            alpha=0.8,
            linewidths=2,
            width=1.5)
    
    plt.title(f'Research Taxonomy Tree - Radial View (Levels 0-{max_level})', 
              fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Radial tree visualization saved to {output_file}")
    plt.close()

def generate_text_tree(taxonomy, indent=0, max_level=None, current_level=0):
    """
    Generate a text-based tree representation.
    """
    tree_lines = []
    prefix = "│   " * indent
    
    if max_level is not None and current_level > max_level:
        return tree_lines
    
    for i, (key, value) in enumerate(taxonomy.items()):
        is_last = (i == len(taxonomy) - 1)
        connector = "└── " if is_last else "├── "
        
        tree_lines.append(f"{prefix}{connector}{key}")
        
        if isinstance(value, dict):
            next_indent = indent + (0 if is_last else 1)
            subtree = generate_text_tree(value, next_indent, max_level, current_level + 1)
            tree_lines.extend(subtree)
        elif isinstance(value, list) and len(value) > 0:
            # Show first few items from the list
            sub_prefix = "    " * (indent + 1)
            for j, item in enumerate(value[:5]):  # Show first 5 items
                item_connector = "    └── " if j == min(4, len(value) - 1) else "    ├── "
                tree_lines.append(f"{sub_prefix}{item_connector}{item}")
            if len(value) > 5:
                tree_lines.append(f"{sub_prefix}    └── ... ({len(value) - 5} more)")
    
    return tree_lines

def save_text_tree(taxonomy, output_file='taxonomy_tree.txt', max_level=None):
    """
    Save a text-based tree to a file.
    """
    tree_lines = ["Research Taxonomy Structure", "=" * 50, ""]
    tree_lines.extend(generate_text_tree(taxonomy, max_level=max_level))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tree_lines))
    
    print(f"Text tree saved to {output_file}")

def create_domain_trees(taxonomy, output_dir='.'):
    """
    Create separate tree visualizations for each major domain.
    """
    import os
    
    for domain_name, domain_content in taxonomy.items():
        if isinstance(domain_content, dict):
            output_file = os.path.join(output_dir, f'taxonomy_tree_{domain_name.replace(" ", "_").replace("/", "_")}.png')
            
            # Build graph for this domain only
            G = build_graph({domain_name: domain_content}, max_level=4)
            
            # Create figure
            plt.figure(figsize=(24, 16))
            
            # Use hierarchical layout
            pos = hierarchy_pos(G, width=2.5, vert_gap=0.18)
            
            # Remove dummy root if added
            if "ROOT" in G.nodes():
                G.remove_node("ROOT")
                del pos["ROOT"]
            
            # Get node levels
            levels = nx.get_node_attributes(G, 'level')
            colors = [levels[node] for node in G.nodes()]
            
            # Draw
            nx.draw(G, pos,
                    node_color=colors,
                    node_size=1500,
                    cmap=plt.cm.Set3,
                    with_labels=True,
                    font_size=7,
                    font_weight='bold',
                    arrows=True,
                    edge_color='gray',
                    alpha=0.9,
                    arrowsize=10,
                    linewidths=2)
            
            plt.title(f'{domain_name} - Taxonomy Tree', fontsize=18, fontweight='bold', pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {output_file}")
            plt.close()

if __name__ == "__main__":
    # Load taxonomy
    taxonomy_file = 'preprocessed_taxonomy.json'
    taxonomy = load_taxonomy(taxonomy_file)
    
    print("Generating taxonomy tree visualizations...")
    print("=" * 60)
    
    # 1. Generate text tree (all levels)
    print("\n1. Generating text-based tree...")
    save_text_tree(taxonomy, 'taxonomy_tree_full.txt', max_level=None)
    save_text_tree(taxonomy, 'taxonomy_tree_overview.txt', max_level=2)
    
    # 2. Generate 2-level hierarchical visualization (main domains + their immediate subcategories)
    print("\n2. Generating 2-level hierarchical tree visualization...")
    visualize_taxonomy_hierarchical(taxonomy, 'taxonomy_tree_2levels.png', max_level=1)
    
    # 3. Generate radial visualization
    print("\n3. Generating radial tree visualization...")
    visualize_taxonomy_radial(taxonomy, 'taxonomy_tree_radial.png', max_level=2)
    
    # 4. Generate separate trees for each major domain
    print("\n4. Generating individual domain trees...")
    create_domain_trees(taxonomy, output_dir='.')
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - taxonomy_tree_full.txt (complete text tree)")
    print("  - taxonomy_tree_overview.txt (text tree, 2 levels)")
    print("  - taxonomy_tree_2levels.png (MAIN TREE: 2 levels - domains + subcategories)")
    print("  - taxonomy_tree_radial.png (radial view, 2 levels)")
    print("  - taxonomy_tree_*.png (individual domain trees)")
