#!/usr/bin/env python3
"""
RAG System Evaluation Script
Compares model accuracy before and after RAG (retrieval + re-ranking)
"""

import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add RAG folder to path
sys.path.insert(0, str(Path(__file__).parent))

from local_model_classifier import LocalModelClassifier
from vector_db_manager import VectorDBManager
from taxonomy_parser import TaxonomyParser
from config import EMBEDDING_MODEL_NAME, TAXONOMY_PATH, CHROMA_DB_PATH


def load_test_data(data_path: str, n_samples: int = 1000) -> pd.DataFrame:
    """Load test data from JSON files"""
    print(f"Loading test data from: {data_path}")
    
    data_files = list(Path(data_path).glob("*.json"))
    all_data = []
    
    for file_path in data_files[:10]:  # Limit files for faster loading
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
        
        if len(all_data) >= n_samples:
            break
    
    # Sample and create DataFrame
    df = pd.DataFrame(all_data[:n_samples])
    
    print(f"✓ Loaded {len(df)} samples")
    print(f"  Columns: {df.columns.tolist()}")
    
    return df


def evaluate_without_rag(
    model: LocalModelClassifier,
    test_df: pd.DataFrame,
    taxonomy_paths: List[str]
) -> Dict:
    """
    Evaluate model WITHOUT RAG (direct classification on all classes)
    Note: This won't work well because we don't have index->path mapping
    """
    print("\n" + "="*80)
    print("EVALUATION WITHOUT RAG (Direct Model Prediction)")
    print("="*80)
    
    results = []
    correct = 0
    total = 0
    
    for idx, row in test_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(test_df)}...")
        
        title = row.get('title', '')
        abstract = row.get('abstract', '')
        true_path = row.get('classification_path', '')
        
        if not title or not abstract or not true_path:
            continue
        
        # Direct model prediction (returns top candidate from all paths)
        # Since we don't have mapping, use top 10 paths as candidates
        result = model.classify_with_paths(
            title=title,
            abstract=abstract,
            candidate_paths=taxonomy_paths[:10]  # Random baseline
        )
        
        predicted_path = result.get('path', '')
        is_correct = predicted_path == true_path
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'title': title[:50],
            'true_path': true_path,
            'predicted_path': predicted_path,
            'correct': is_correct,
            'confidence': result.get('confidence', 0)
        })
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n{'─'*80}")
    print(f"WITHOUT RAG Results:")
    print(f"  • Total samples: {total}")
    print(f"  • Correct predictions: {correct}")
    print(f"  • Accuracy: {accuracy:.2f}%")
    print(f"{'─'*80}")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_prediction_time': 0,  # Not tracked in this version
        'results': results
    }


def evaluate_with_rag(
    model: LocalModelClassifier,
    db_manager: VectorDBManager,
    test_df: pd.DataFrame,
    top_k: int = 5
) -> Dict:
    """
    Evaluate model WITH RAG (retrieval + re-ranking)
    """
    print("\n" + "="*80)
    print("EVALUATION WITH RAG (Retrieval + Re-ranking)")
    print("="*80)
    
    results = []
    correct = 0
    correct_in_retrieved = 0
    total = 0
    retrieval_times = []
    reranking_times = []
    
    # Track accuracy at different positions
    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0
    
    for idx, row in test_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(test_df)}...")
        
        title = row.get('title', '')
        abstract = row.get('abstract', '')
        true_path = row.get('classification_path', '')
        
        if not title or not abstract or not true_path:
            continue
        
        # Step 1: Retrieval
        query_text = f"Title: {title}\n\nAbstract: {abstract}"
        
        start_time = time.time()
        retrieval_result = db_manager.retrieve_relevant_paths(
            query_text=query_text,
            top_k=top_k
        )
        retrieval_time = time.time() - start_time
        retrieval_times.append(retrieval_time)
        
        retrieved_paths = retrieval_result['retrieved_paths']
        
        # Check if true path is in retrieved
        retrieved_path_strings = [p['path'] for p in retrieved_paths]
        true_in_retrieved = true_path in retrieved_path_strings
        if true_in_retrieved:
            correct_in_retrieved += 1
        
        # Step 2: Re-ranking with model
        start_time = time.time()
        result = model.classify_article(
            title=title,
            abstract=abstract,
            relevant_paths=retrieved_paths
        )
        reranking_time = time.time() - start_time
        reranking_times.append(reranking_time)
        
        predicted_path = result['classification']['path']
        is_correct = predicted_path == true_path
        
        if is_correct:
            correct += 1
        
        # Check position in retrieved list
        if true_path in retrieved_path_strings:
            pos = retrieved_path_strings.index(true_path)
            if pos == 0:
                top_1_correct += 1
            if pos < 3:
                top_3_correct += 1
            if pos < 5:
                top_5_correct += 1
        
        total += 1
        
        results.append({
            'title': title[:50],
            'true_path': true_path,
            'predicted_path': predicted_path,
            'correct': is_correct,
            'true_in_retrieved': true_in_retrieved,
            'model_score': result['classification'].get('model_score', 0),
            'retrieval_score': result['classification'].get('retrieval_score', 0),
            'combined_score': result['classification'].get('confidence_score', 0),
            'retrieval_time': retrieval_time,
            'reranking_time': reranking_time
        })
    
    accuracy = (correct / total * 100) if total > 0 else 0
    retrieval_recall = (correct_in_retrieved / total * 100) if total > 0 else 0
    
    print(f"\n{'─'*80}")
    print(f"WITH RAG Results:")
    print(f"  • Total samples: {total}")
    print(f"  • Correct final predictions: {correct}")
    print(f"  • Final Accuracy: {accuracy:.2f}%")
    print(f"\n  Retrieval Performance:")
    print(f"  • Retrieval Recall@{top_k}: {retrieval_recall:.2f}%")
    print(f"  • Top-1 in retrieved: {top_1_correct}/{total} ({top_1_correct/total*100:.2f}%)")
    print(f"  • Top-3 in retrieved: {top_3_correct}/{total} ({top_3_correct/total*100:.2f}%)")
    print(f"  • Top-5 in retrieved: {top_5_correct}/{total} ({top_5_correct/total*100:.2f}%)")
    print(f"\n  Timing:")
    print(f"  • Avg retrieval time: {np.mean(retrieval_times):.4f}s")
    print(f"  • Avg re-ranking time: {np.mean(reranking_times):.4f}s")
    print(f"  • Total avg time: {np.mean(retrieval_times) + np.mean(reranking_times):.4f}s")
    print(f"{'─'*80}")
    
    return {
        'accuracy': accuracy,
        'retrieval_recall': retrieval_recall,
        'correct': correct,
        'total': total,
        'correct_in_retrieved': correct_in_retrieved,
        'top_1_correct': top_1_correct,
        'top_3_correct': top_3_correct,
        'top_5_correct': top_5_correct,
        'avg_retrieval_time': np.mean(retrieval_times),
        'avg_reranking_time': np.mean(reranking_times),
        'results': results
    }


def analyze_embeddings(db_manager: VectorDBManager, test_df: pd.DataFrame, n_samples: int = 10):
    """Analyze embedding quality and similarity distributions"""
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    similarities = []
    
    for idx, row in test_df.head(n_samples).iterrows():
        title = row.get('title', '')
        abstract = row.get('abstract', '')
        true_path = row.get('classification_path', '')
        
        if not title or not abstract:
            continue
        
        query_text = f"Title: {title}\n\nAbstract: {abstract}"
        
        # Get embedding
        embedding = db_manager.embedding_model.encode(query_text)
        
        # Retrieve
        retrieval_result = db_manager.retrieve_relevant_paths(query_text=query_text, top_k=5)
        
        print(f"\nSample {idx + 1}:")
        print(f"  Title: {title[:60]}...")
        print(f"  True path: {true_path[:80]}...")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"  Top 5 retrieved:")
        
        for i, result in enumerate(retrieval_result['retrieved_paths'][:5], 1):
            is_correct = "✓" if result['path'] == true_path else " "
            print(f"    {is_correct} {i}. {result['path'][:70]}... (sim: {result['similarity']:.4f})")
            similarities.append(result['similarity'])
    
    print(f"\n{'─'*80}")
    print(f"Similarity Statistics (from {n_samples} samples):")
    print(f"  • Mean: {np.mean(similarities):.4f}")
    print(f"  • Std: {np.std(similarities):.4f}")
    print(f"  • Min: {np.min(similarities):.4f}")
    print(f"  • Max: {np.max(similarities):.4f}")
    print(f"{'─'*80}")
    
    return similarities


def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("RAG SYSTEM EVALUATION")
    print("="*80)
    
    # Configuration
    test_data_path = "/home/zahra/Documents/4rth Year/NLP/Project/final_data/final_nlp_data/final_final_annotation_test_data_fast"
    n_samples = 300
    top_k = 5
    
    # Load test data
    test_df = load_test_data(test_data_path, n_samples)
    
    # Initialize components
    print("\nInitializing RAG components...")
    
    # Taxonomy
    parser = TaxonomyParser(TAXONOMY_PATH)
    taxonomy_paths_list = parser.extract_all_paths()
    # Handle both dict and string formats
    if taxonomy_paths_list and isinstance(taxonomy_paths_list[0], dict):
        taxonomy_path_strings = [p.get('path', str(p)) for p in taxonomy_paths_list]
    else:
        taxonomy_path_strings = [str(p) for p in taxonomy_paths_list]
    
    # Vector DB
    db_manager = VectorDBManager(
        db_path=CHROMA_DB_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    db_manager.initialize_collection(reset=False)
    
    # Local Model - Using LocalModelClassifier with LoRA (862 labels)
    model = LocalModelClassifier()
    
    print("✓ All components initialized\n")
    
    # Analyze embeddings
    analyze_embeddings(db_manager, test_df, n_samples=10)
    
    # Skip WITHOUT RAG evaluation
    # direct_results = evaluate_without_rag(model, test_df, taxonomy_path_strings)
    
    # Evaluate WITH RAG
    rag_results = evaluate_with_rag(model, db_manager, test_df, top_k=top_k)
    
    # Save detailed results
    output_dir = Path(__file__).parent / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save summary
    summary = {
        'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': n_samples,
        'top_k': top_k,
        'with_rag': {
            'accuracy': rag_results['accuracy'],
            'retrieval_recall': rag_results['retrieval_recall'],
            'top_1_accuracy': rag_results['top_1_correct'] / rag_results['total'] * 100,
            'top_3_accuracy': rag_results['top_3_correct'] / rag_results['total'] * 100,
            'top_5_accuracy': rag_results['top_5_correct'] / rag_results['total'] * 100,
            'avg_retrieval_time': rag_results['avg_retrieval_time'],
            'avg_reranking_time': rag_results['avg_reranking_time'],
        }
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    pd.DataFrame(rag_results['results']).to_csv(
        output_dir / 'detailed_results_with_rag.csv', 
        index=False
    )
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  • evaluation_summary.json")
    print(f"  • detailed_results_with_rag.csv")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
