#!/usr/bin/env python3
"""
RAG Pipeline Demo with Detailed Output for Report
This script demonstrates the RAG classification pipeline with verbose output
showing embeddings, retrieved paths, and evaluation metrics.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add RAG folder to path
sys.path.insert(0, str(Path(__file__).parent))

from vector_db_manager import VectorDBManager
from taxonomy_parser import TaxonomyParser
from llm_classifier import LLMClassifier
from config import EMBEDDING_MODEL_NAME, TAXONOMY_PATH, CHROMA_DB_PATH, RAG_CONFIG


def print_header(title: str, char: str = "="):
    """Print a formatted header"""
    width = 80
    print("\n" + char * width)
    print(f" {title} ".center(width))
    print(char * width + "\n")


def print_subheader(title: str):
    """Print a formatted subheader"""
    print(f"\n{'─' * 60}")
    print(f"► {title}")
    print(f"{'─' * 60}\n")


def print_embedding_info(db_manager, query_text: str):
    """Print embedding details"""
    print_subheader("EMBEDDING GENERATION")
    
    model = db_manager.embedding_model
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Model Max Sequence Length: {model.max_seq_length}")
    print(f"Embedding Dimension: {model.get_sentence_embedding_dimension()}")
    
    # Generate embedding for the query
    start_time = time.time()
    embedding = model.encode(query_text, convert_to_numpy=True)
    embed_time = time.time() - start_time
    
    print(f"\nQuery Text Length: {len(query_text)} characters")
    print(f"Embedding Generation Time: {embed_time:.4f} seconds")
    print(f"Embedding Shape: {embedding.shape}")
    print(f"Embedding L2 Norm: {np.linalg.norm(embedding):.4f}")
    print(f"\nEmbedding Vector (first 10 dimensions):")
    print(f"  {embedding[:10].round(4).tolist()}")
    print(f"  ... ({len(embedding) - 10} more dimensions)")
    
    return embedding


def print_retrieval_results(db_manager, title: str, abstract: str, top_k: int = 5):
    """Print detailed retrieval results"""
    print_subheader("VECTOR SIMILARITY SEARCH (RETRIEVAL)")
    
    query_text = f"Title: {title}\n\nAbstract: {abstract}"
    
    # Perform retrieval with timing
    start_time = time.time()
    retrieval_result = db_manager.retrieve_relevant_paths(
        query_text=query_text,
        top_k=top_k,
        similarity_threshold=RAG_CONFIG.get('similarity_threshold')
    )
    retrieval_time = time.time() - start_time
    
    print(f"Retrieval Parameters:")
    print(f"  • Top-K: {top_k}")
    print(f"  • Similarity Threshold: {RAG_CONFIG.get('similarity_threshold', 'None')}")
    print(f"  • Collection Size: {db_manager.collection.count()} paths")
    print(f"  • Retrieval Time: {retrieval_time:.4f} seconds")
    
    print(f"\n✓ Retrieved {len(retrieval_result['retrieved_paths'])} Candidate Paths:")
    print("-" * 60)
    
    for i, path_info in enumerate(retrieval_result['retrieved_paths'], 1):
        print(f"\n  Rank #{i}:")
        print(f"    Path: {path_info['path']}")
        print(f"    Domain: {path_info['domain']}")
        print(f"    Field: {path_info['field']}")
        print(f"    Level: {path_info['level']}")
        print(f"    Cosine Similarity: {path_info['similarity']:.4f}")
        print(f"    Distance: {path_info['distance']:.4f}")
    
    return retrieval_result, retrieval_time


def print_evaluation_summary(results: list):
    """Print evaluation metrics summary"""
    print_subheader("RETRIEVAL EVALUATION SUMMARY")
    
    total = len(results)
    print(f"Total Articles Processed: {total}")
    
    # Calculate average retrieval times
    avg_retrieval = np.mean([r['retrieval_time'] for r in results])
    avg_llm = np.mean([r.get('llm_time', 0) for r in results])
    print(f"\nAverage Processing Times:")
    print(f"  • Embedding + Retrieval: {avg_retrieval:.4f}s")
    print(f"  • LLM Classification: {avg_llm:.4f}s")
    print(f"  • Total per Article: {avg_retrieval + avg_llm:.4f}s")
    
    # Similarity score distribution
    all_similarities = []
    for r in results:
        if r['retrieved_paths']:
            sims = [p['similarity'] for p in r['retrieved_paths']]
            all_similarities.extend(sims)
    
    if all_similarities:
        print(f"\nRetrieval Similarity Score Distribution:")
        print(f"  • Mean: {np.mean(all_similarities):.4f}")
        print(f"  • Std Dev: {np.std(all_similarities):.4f}")
        print(f"  • Min: {np.min(all_similarities):.4f}")
        print(f"  • Max: {np.max(all_similarities):.4f}")
        
        # Top-1 statistics
        top1_sims = [r['retrieved_paths'][0]['similarity'] for r in results if r['retrieved_paths']]
        print(f"\nTop-1 Retrieval Statistics:")
        print(f"  • Mean Top-1 Similarity: {np.mean(top1_sims):.4f}")
        print(f"  • Min Top-1 Similarity: {np.min(top1_sims):.4f}")
    
    # LLM Classification Summary
    print(f"\nLLM Classification Summary:")
    confidence_counts = {'High': 0, 'Medium': 0, 'Low': 0, 'Unknown': 0}
    for r in results:
        conf = r.get('llm_confidence', 'Unknown')
        if conf in confidence_counts:
            confidence_counts[conf] += 1
        else:
            confidence_counts['Unknown'] += 1
    
    for level, count in confidence_counts.items():
        if count > 0:
            print(f"  • {level} Confidence: {count}/{total} ({count/total*100:.1f}%)")


def print_llm_classification(llm_classifier, title: str, abstract: str, retrieved_paths: list):
    """Print LLM classification results with timing"""
    print_subheader("LLM CLASSIFICATION (FINAL DECISION)")
    
    # Extract just the path strings for the LLM
    candidate_paths = [p['path'] for p in retrieved_paths]
    
    print(f"LLM Model: {llm_classifier.model_name}")
    print(f"Number of Candidate Paths: {len(candidate_paths)}")
    print(f"\nCandidate Paths for LLM:")
    for i, path in enumerate(candidate_paths, 1):
        print(f"  {i}. {path}")
    
    # Generate prompt
    prompt = llm_classifier.create_classification_prompt(
        title=title,
        abstract=abstract,
        relevant_paths=candidate_paths,
        include_reasoning=True
    )
    
    print(f"\n{'─' * 40}")
    print(f"PROMPT SENT TO LLM:")
    print(f"{'─' * 40}")
    print(prompt)
    print(f"{'─' * 40}")
    print(f"Prompt Length: {len(prompt)} characters")
    
    # Call LLM with timing
    print(f"\n⏳ Calling LLM API...")
    start_time = time.time()
    
    try:
        response = llm_classifier.classify(
            prompt=prompt,
            temperature=0.1,
            max_tokens=150
        )
        llm_time = time.time() - start_time
        
        print(f"✓ LLM Response received in {llm_time:.4f} seconds")
        
        print(f"\n{'─' * 40}")
        print(f"RAW LLM RESPONSE:")
        print(f"{'─' * 40}")
        print(response)
        print(f"{'─' * 40}")
        print(f"Response Length: {len(response)} characters")
        
        # Parse the response
        parsed = llm_classifier.parse_classification_response(response)
        
        print(f"\n✓ PARSED CLASSIFICATION RESULT:")
        print(f"  • Selected Path: {parsed['path']}")
        print(f"  • Confidence: {parsed['confidence']}")
        if parsed['reasoning']:
            print(f"  • Reasoning: {parsed['reasoning']}")
        
        return {
            'llm_time': llm_time,
            'selected_path': parsed['path'],
            'confidence': parsed['confidence'],
            'reasoning': parsed['reasoning'],
            'raw_response': response
        }
        
    except Exception as e:
        llm_time = time.time() - start_time
        print(f"✗ LLM Error after {llm_time:.4f}s: {e}")
        return {
            'llm_time': llm_time,
            'selected_path': None,
            'confidence': None,
            'reasoning': None,
            'error': str(e)
        }


def run_demo():
    """Run the complete demo with verbose output"""
    print_header("RAG-BASED RESEARCH ARTICLE CLASSIFICATION DEMO", "█")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    print_subheader("SYSTEM CONFIGURATION")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Top-K Retrieval: {RAG_CONFIG['top_k']}")
    print(f"Similarity Threshold: {RAG_CONFIG.get('similarity_threshold', 'None')}")
    print(f"ChromaDB Path: {CHROMA_DB_PATH}")
    
    # Initialize components
    print_subheader("COMPONENT INITIALIZATION")
    print("Loading taxonomy and vector database...")
    
    # Parse taxonomy
    start_time = time.time()
    parser = TaxonomyParser(TAXONOMY_PATH)
    taxonomy_paths = parser.extract_all_paths()
    parse_time = time.time() - start_time
    
    tax_stats = parser.get_statistics()
    print(f"✓ Taxonomy parsed in {parse_time:.2f}s")
    print(f"  • Total paths: {tax_stats['total_paths']}")
    print(f"  • Domains: {tax_stats['domains']}")
    print(f"  • Max level: {tax_stats.get('max_level', 'N/A')}")
    
    # Initialize vector DB
    start_time = time.time()
    db_manager = VectorDBManager(
        db_path=CHROMA_DB_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    db_manager.initialize_collection(reset=False)
    
    # Populate if empty
    if db_manager.collection.count() == 0:
        print("Populating vector database with taxonomy paths...")
        db_manager.populate_from_paths(taxonomy_paths)
    
    db_init_time = time.time() - start_time
    
    print(f"✓ Vector database initialized in {db_init_time:.2f}s")
    print(f"  • Collection size: {db_manager.collection.count()} paths")
    print(f"  • Embedding dimension: {db_manager.embedding_model.get_sentence_embedding_dimension()}")
    
    # Initialize LLM Classifier
    print_subheader("LLM CLASSIFIER INITIALIZATION")
    try:
        llm_classifier = LLMClassifier(model_name="gemini-2.0-flash")
        print(f"✓ LLM Classifier initialized")
        print(f"  • Model: {llm_classifier.model_name}")
        print(f"  • API: Google AI Studio (Gemini)")
        llm_available = True
    except Exception as e:
        print(f"✗ LLM Classifier failed to initialize: {e}")
        print(f"  • Will skip LLM classification step")
        llm_classifier = None
        llm_available = False
    
    # Example articles for demonstration
    demo_articles = [
        {
            'title': 'Artificial Intelligence in Education and Schools',
            'abstract': 'With the increase in studies about artificial intelligence (AI) in the educational field, many scholars in the field believe '
                       'that the role of teachers, school and leaders in education will change. In this regard, the purpose of this study is to '
                       'examine what possible scenarios are there with the arrival of AI in education and what kind of implications it can reveal '
                       'for future of schools. The research was designed as a phenomenological study, a qualitative research method, in which '
                       'the opinions of participants from different sectors were examined. The results show that schools and teachers will have '
                       'new products, benefits and also face drawbacks with the arrival of AI in education. The findings point out some '
                       'suggestions for use of AI and prevention of possible problems. While participants generally seem to have positive '
                       'perceptions towards AI, there are also certain drawbacks, especially highlighted by teachers and academicians, '
                       'regarding the future of teaching. Lawyers and jurists tend to focus more on legal grounds for AI in education and future '
                       'problems, while engineers see AI as a tool to bring quality and benefit for all in the education sector.'
        },
        {
            'title': 'Deep Learning for Medical Image Segmentation: A Comprehensive Review',
            'abstract': 'This paper presents a comprehensive review of deep learning techniques '
                       'for medical image segmentation. We analyze convolutional neural networks (CNNs), '
                       'U-Net architectures, and transformer-based models for segmenting anatomical '
                       'structures in CT, MRI, and X-ray images. Our experiments on public datasets '
                       'demonstrate state-of-the-art performance in tumor detection and organ segmentation.'
        },
        {
            'title': 'Climate Change Impact on Agricultural Productivity in Sub-Saharan Africa',
            'abstract': 'We investigate the effects of climate change on crop yields across '
                       'Sub-Saharan Africa using satellite imagery and statistical models. Our analysis '
                       'covers maize, wheat, and sorghum production from 1990 to 2023. Results indicate '
                       'significant yield reductions due to increased temperature and drought frequency, '
                       'with implications for food security policy.'
        },
        {
            'title': 'Quantum Error Correction in Superconducting Qubits',
            'abstract': 'We demonstrate a novel quantum error correction protocol for superconducting '
                       'qubit systems. Using surface codes and real-time feedback, we achieve error rates '
                       'below the fault-tolerance threshold. Our implementation on a 17-qubit processor '
                       'shows promising results for scalable quantum computing applications.'
        }
    ]
    
    # Process each article
    all_results = []
    
    for i, article in enumerate(demo_articles, 1):
        print_header(f"ARTICLE {i} OF {len(demo_articles)}", "═")
        
        print(f"Title: {article['title']}")
        print(f"\nAbstract: {article['abstract'][:200]}...")
        
        # Show embedding info
        query_text = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
        print_embedding_info(db_manager, query_text)
        
        # Show retrieval results
        retrieval_result, retrieval_time = print_retrieval_results(
            db_manager, 
            article['title'], 
            article['abstract'],
            top_k=RAG_CONFIG['top_k']
        )
        
        # Store results for evaluation
        all_results.append({
            'title': article['title'],
            'retrieval_time': retrieval_time,
            'retrieved_paths': retrieval_result['retrieved_paths'],
            'llm_time': 0,
            'llm_confidence': None,
            'llm_selected_path': None
        })
        
        # LLM Classification Step
        if llm_available and retrieval_result['retrieved_paths']:
            llm_result = print_llm_classification(
                llm_classifier,
                article['title'],
                article['abstract'],
                retrieval_result['retrieved_paths']
            )
            
            # Update results with LLM info
            all_results[-1]['llm_time'] = llm_result['llm_time']
            all_results[-1]['llm_confidence'] = llm_result['confidence']
            all_results[-1]['llm_selected_path'] = llm_result['selected_path']
        
        # Show interpretation
        print_subheader("FINAL CLASSIFICATION RESULT")
        if retrieval_result['retrieved_paths']:
            if llm_available and all_results[-1]['llm_selected_path']:
                print(f"✓ LLM Final Decision:")
                print(f"  • Selected Path: {all_results[-1]['llm_selected_path']}")
                print(f"  • Confidence: {all_results[-1]['llm_confidence']}")
                print(f"\n  Pipeline Summary:")
                print(f"  • Retrieval found {len(retrieval_result['retrieved_paths'])} candidates")
                print(f"  • LLM selected the best matching path")
                print(f"  • Total time: {retrieval_time + all_results[-1]['llm_time']:.4f}s")
            else:
                top_path = retrieval_result['retrieved_paths'][0]
                print(f"Best Match (Retrieval Only):")
                print(f"  The article \"{article['title'][:50]}...\"")
                print(f"  was matched to the domain: {top_path['domain']}")
                print(f"  with field: {top_path['field']}")
                print(f"  at similarity score: {top_path['similarity']:.4f}")
        
        # Add separator between articles
        if i < len(demo_articles):
            print("\n" + "▼" * 80 + "\n")
    
    # Final evaluation summary
    print_header("FINAL EVALUATION SUMMARY", "█")
    print_evaluation_summary(all_results)
    
    # System performance summary
    print_subheader("SYSTEM PERFORMANCE SUMMARY")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Vector Database: ChromaDB with cosine similarity")
    print(f"LLM Model: gemini-2.0-flash" if llm_available else "LLM: Not available")
    print(f"Total Taxonomy Paths: {db_manager.collection.count()}")
    
    total_retrieval_time = sum(r['retrieval_time'] for r in all_results)
    total_llm_time = sum(r.get('llm_time', 0) for r in all_results)
    total_time = total_retrieval_time + total_llm_time
    
    print(f"\nTotal Processing Times for {len(demo_articles)} articles:")
    print(f"  • Retrieval (embedding + search): {total_retrieval_time:.4f}s")
    print(f"  • LLM Classification: {total_llm_time:.4f}s")
    print(f"  • Total Pipeline Time: {total_time:.4f}s")
    print(f"\nAverage Time per Article:")
    print(f"  • Retrieval: {total_retrieval_time/len(demo_articles):.4f}s")
    print(f"  • LLM: {total_llm_time/len(demo_articles):.4f}s")
    print(f"  • Total: {total_time/len(demo_articles):.4f}s")
    
    print_header("DEMO COMPLETE", "█")
    print(f"Results can be used for report documentation.")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    results = run_demo()
