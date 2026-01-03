#!/usr/bin/env python3
"""
Generate figures for the RAG evaluation report
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def create_performance_comparison_chart():
    """
    Figure 2: Performance Comparison Bar Chart
    Baseline (No RAG): 27%
    Proposed (With RAG): 34.54%
    Pure Retrieval (Top-1): 20%
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    systems = ['Pure Retrieval\n(Top-1)', 'Baseline\n(No RAG)', 'Proposed\n(With RAG)']
    accuracies = [20.00, 27.00, 34.54]
    colors = ['#95a5a6', '#3498db', '#27ae60']
    
    # Create bars
    bars = ax.bar(systems, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement annotations
    # Arrow from Baseline to Proposed
    ax.annotate('', xy=(2, 27), xytext=(2, 34.54),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(2.15, 30.5, '+7.54%\nimprovement', 
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))
    
    # Arrow from Pure Retrieval to Proposed
    ax.annotate('', xy=(0, 20), xytext=(2, 20),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5, linestyle='--'))
    ax.text(1, 21, '+14.54%', fontsize=9, color='green', ha='center')
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('System Configuration', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: RAG vs Baseline', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 42)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add horizontal line for baseline
    ax.axhline(y=27, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('figure2_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figure2_performance_comparison.png")
    plt.close()


def create_system_architecture_flowchart():
    """
    Figure 1: System Architecture Flowchart
    Shows: Input → Embedding → Vector DB → Top-K Candidates → SciBERT Re-ranker → Final Prediction
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    color_input = '#e8f4f8'
    color_retrieval = '#d6eaf8'
    color_rerank = '#fdebd0'
    color_output = '#d5f4e6'
    
    # Title
    ax.text(5, 9.5, 'RAG System Architecture', fontsize=16, fontweight='bold', 
            ha='center', va='top')
    
    # ==================== INPUT ====================
    # Input box
    input_box = FancyBboxPatch((1, 8), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_input, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 8.4, 'Input Article', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.5, 8.15, '(Title + Abstract)', fontsize=8, ha='center', style='italic')
    
    # Arrow down
    arrow1 = FancyArrowPatch((2.5, 8), (2.5, 7.3),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # ==================== STAGE 1: RETRIEVAL ====================
    # Stage 1 label
    ax.text(0.3, 6.5, 'Stage 1:', fontsize=10, fontweight='bold', color='#2874a6')
    ax.text(0.3, 6.2, 'Semantic\nRetrieval', fontsize=8, color='#2874a6')
    
    # Embedding box
    embed_box = FancyBboxPatch((1.2, 6.5), 2.6, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=color_retrieval,
                               edgecolor='#2874a6', linewidth=2)
    ax.add_patch(embed_box)
    ax.text(2.5, 6.95, 'Encode Query', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.5, 6.65, 'all-MiniLM-L6-v2', fontsize=8, ha='center')
    ax.text(2.5, 6.45, '(384 dims)', fontsize=7, ha='center', style='italic')
    
    # Arrow down
    arrow2 = FancyArrowPatch((2.5, 6.5), (2.5, 5.8),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#2874a6')
    ax.add_patch(arrow2)
    
    # Vector DB box
    db_box = FancyBboxPatch((1.2, 4.7), 2.6, 1.1,
                            boxstyle="round,pad=0.1",
                            facecolor=color_retrieval,
                            edgecolor='#2874a6', linewidth=2)
    ax.add_patch(db_box)
    ax.text(2.5, 5.5, 'Vector Database', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.5, 5.25, 'ChromaDB', fontsize=9, ha='center')
    ax.text(2.5, 5.0, '1,449 taxonomy paths', fontsize=8, ha='center')
    ax.text(2.5, 4.8, 'Cosine similarity', fontsize=7, ha='center', style='italic')
    
    # Arrow down
    arrow3 = FancyArrowPatch((2.5, 4.7), (2.5, 4.0),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#2874a6')
    ax.add_patch(arrow3)
    
    # Top-K box
    topk_box = FancyBboxPatch((1.2, 3.2), 2.6, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=color_retrieval,
                              edgecolor='#2874a6', linewidth=2)
    ax.add_patch(topk_box)
    ax.text(2.5, 3.65, 'Top-K Candidates', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.5, 3.4, 'K = 5 taxonomy paths', fontsize=8, ha='center')
    
    # Arrow right
    arrow4 = FancyArrowPatch((3.8, 3.6), (5.2, 3.6),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2.5, color='black')
    ax.add_patch(arrow4)
    
    # ==================== STAGE 2: RE-RANKING ====================
    # Stage 2 label
    ax.text(5.2, 6.5, 'Stage 2:', fontsize=10, fontweight='bold', color='#d68910')
    ax.text(5.2, 6.2, 'Neural\nRe-ranking', fontsize=8, color='#d68910')
    
    # SciBERT box
    scibert_box = FancyBboxPatch((5.2, 4.7), 3.6, 1.3,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color_rerank,
                                 edgecolor='#d68910', linewidth=2)
    ax.add_patch(scibert_box)
    ax.text(7.0, 5.7, 'Fine-tuned SciBERT', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.0, 5.45, 'with LoRA', fontsize=9, ha='center')
    ax.text(7.0, 5.2, 'Input: [Article] [SEP] [Path]', fontsize=8, ha='center', style='italic')
    ax.text(7.0, 4.95, 'Score each candidate', fontsize=8, ha='center')
    
    # Score fusion box
    fusion_box = FancyBboxPatch((5.2, 3.2), 3.6, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=color_rerank,
                                edgecolor='#d68910', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(7.0, 3.7, 'Score Fusion', fontsize=10, fontweight='bold', ha='center')
    ax.text(7.0, 3.4, '0.6 × Model + 0.4 × Retrieval', fontsize=8, ha='center', 
            family='monospace', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Arrow down
    arrow5 = FancyArrowPatch((7.0, 3.2), (7.0, 2.5),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow5)
    
    # ==================== OUTPUT ====================
    # Final prediction box
    output_box = FancyBboxPatch((5.5, 1.5), 3.0, 1.0,
                                boxstyle="round,pad=0.1",
                                facecolor=color_output,
                                edgecolor='#0b5345', linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(7.0, 2.2, 'Final Classification', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.0, 1.95, 'Best taxonomy path', fontsize=9, ha='center')
    ax.text(7.0, 1.7, '(Highest combined score)', fontsize=7, ha='center', style='italic')
    
    # Add timing annotations
    ax.text(4.5, 5.5, '~0.024s', fontsize=8, color='#2874a6', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2874a6'))
    ax.text(4.5, 3.8, '~0.830s', fontsize=8, color='#d68910',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d68910'))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=color_retrieval, edgecolor='#2874a6', label='Retrieval Stage'),
        mpatches.Patch(facecolor=color_rerank, edgecolor='#d68910', label='Re-ranking Stage'),
        mpatches.Patch(facecolor=color_output, edgecolor='#0b5345', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, frameon=True)
    
    plt.tight_layout()
    plt.savefig('figure1_system_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figure1_system_architecture.png")
    plt.close()


if __name__ == "__main__":
    print("Generating figures for RAG evaluation report...")
    print("=" * 60)
    
    # Generate figures
    create_system_architecture_flowchart()
    create_performance_comparison_chart()
    
    print("=" * 60)
    print("✓ All figures generated successfully!")
    print("\nGenerated files:")
    print("  - figure1_system_architecture.png")
    print("  - figure2_performance_comparison.png")
