"""
Final Model Classifier - Loads the full fine-tuned SciBERT model
This model was trained directly on 1393 labels without LoRA adapters.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Optional
import os


class FinalModelClassifier:
    """
    Classifier using the full fine-tuned SciBERT model (1393 labels)
    Model Location: RAG/final_model/
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier with the fine-tuned model
        
        Args:
            model_path: Path to the fine-tuned model folder (default: RAG/final_model/)
        """
        if model_path is None:
            # Default to final_model in RAG directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "final_model")
        
        self.model_path = model_path
        self.device = torch.device('cpu')  # Use CPU to avoid CUDA issues
        
        print(f"Loading final fine-tuned model from: {model_path}")
        
        # Load the checkpoint
        checkpoint_file = os.path.join(model_path, "best_model.pt")
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        
        # Extract metadata from label_encoders
        # This is a hierarchical model with multiple levels
        # Use the last encoder which has the leaf-level labels (1393 labels)
        if 'label_encoders' in checkpoint and isinstance(checkpoint['label_encoders'], list):
            # Use the last encoder (leaf level) - index -1 or 6
            encoder_info = checkpoint['label_encoders'][-1]  # Last encoder for leaf labels
            self.label_to_id = encoder_info.get('label_to_id', {})
            self.id_to_label = encoder_info.get('id_to_label', {})
            self.num_labels = encoder_info.get('num_labels', len(self.id_to_label))
            
            print(f"✓ Loaded hierarchical model with {len(checkpoint['label_encoders'])} levels")
            print(f"  Using leaf level (Level {len(checkpoint['label_encoders'])-1}) with {self.num_labels} labels")
        else:
            raise ValueError("Could not find label_encoders in checkpoint")
        
        print(f"F1 Score from training: {checkpoint.get('f1_score', 'N/A'):.4f}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        
        # Load model architecture with correct num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "allenai/scibert_scivocab_uncased",
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Load the trained weights from the checkpoint
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print("Warning: Could not find model_state_dict in checkpoint")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully")
    
    def classify_with_paths(
        self,
        title: str,
        abstract: str,
        candidate_paths: List[str] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Classify article using re-ranking with the fine-tuned model
        
        Args:
            title: Article title
            abstract: Article abstract
            candidate_paths: List of candidate taxonomy paths to re-rank
            top_k: Number of top predictions to return (default: 5)
        
        Returns:
            Dictionary with classification results
        """
        # Combine title and abstract
        article_text = f"{title} {abstract}"
        
        # If no candidates provided, return top-k predictions from the model
        if candidate_paths is None or len(candidate_paths) == 0:
            return self._classify_direct(article_text, top_k)
        
        # Re-ranking mode: score each candidate path
        results = []
        
        for path in candidate_paths:
            # Create combined input: [article] [SEP] [path]
            combined_text = f"{article_text} [SEP] {path}"
            
            # Tokenize
            inputs = self.tokenizer(
                combined_text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Use max probability as relevance score
                model_score = float(torch.max(probs).cpu().numpy())
            
            results.append({
                'path': path,
                'model_score': model_score
            })
        
        # Sort by model score (descending)
        results.sort(key=lambda x: x['model_score'], reverse=True)
        
        # Return top result
        if results:
            best_result = results[0]
            return {
                'path': best_result['path'],
                'confidence': best_result['model_score'],
                'model_score': best_result['model_score'],
                'all_scores': results[:top_k]
            }
        
        return {'path': '', 'confidence': 0.0}
    
    def _classify_direct(self, article_text: str, top_k: int = 5) -> Dict:
        """
        Direct classification without candidate paths
        Returns top-k predictions from the model
        """
        # Tokenize
        inputs = self.tokenizer(
            article_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(probs[0])))
        
        # Convert to label names
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            label_id = int(idx.cpu().numpy())
            label_name = self.id_to_label.get(label_id, f"Unknown_{label_id}")
            predictions.append({
                'label': label_name,
                'confidence': float(prob.cpu().numpy()),
                'label_id': label_id
            })
        
        # Return top prediction
        if predictions:
            return {
                'path': predictions[0]['label'],
                'confidence': predictions[0]['confidence'],
                'top_predictions': predictions
            }
        
        return {'path': '', 'confidence': 0.0}
    
    def classify(self, title: str, abstract: str) -> Dict:
        """
        Simple classification interface - returns single best prediction
        """
        result = self._classify_direct(f"{title} {abstract}", top_k=1)
        return {
            'predicted_label': result.get('path', ''),
            'confidence': result.get('confidence', 0.0)
        }
    
    def classify_article(
        self,
        title: str,
        abstract: str,
        relevant_paths: List[str],
        retrieval_similarities: Optional[List[float]] = None,
        **kwargs
    ) -> Dict:
        """
        Complete classification compatible with evaluation script interface
        
        Uses the model to re-rank the top retrieved paths
        
        Args:
            title: Article title
            abstract: Article abstract
            relevant_paths: Retrieved taxonomy paths
            retrieval_similarities: Similarity scores from retrieval (optional)
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            Classification result with metadata
        """
        # Extract similarities from retrieved paths if provided as dicts
        if relevant_paths and isinstance(relevant_paths[0], dict):
            retrieval_similarities = [p.get('similarity', 0.0) for p in relevant_paths]
            relevant_paths = [p.get('path', p) for p in relevant_paths]
        
        # Run classification with re-ranking
        result = self.classify_with_paths(
            title=title,
            abstract=abstract,
            candidate_paths=relevant_paths,
            top_k=len(relevant_paths)
        )
        
        # Format response to match expected interface
        confidence_label = "High" if result['confidence'] > 0.7 else \
                          "Medium" if result['confidence'] > 0.5 else "Low"
        
        reasoning = f"Re-ranked {len(relevant_paths)} candidates using fine-tuned model (1393 labels). " \
                   f"Model score: {result.get('model_score', 0):.3f}"
        
        return {
            'classification': {
                'path': result['path'],
                'confidence': confidence_label,
                'confidence_score': result['confidence'],
                'model_score': result.get('model_score', 0),
                'reasoning': reasoning,
                'raw_response': f"Re-ranking with score: {result['confidence']:.4f}"
            },
            'prompt_length': 0,  # Not applicable for local model
            'response_length': 0,
        }


# Test the classifier
if __name__ == "__main__":
    print("Testing Final Model Classifier...")
    print("=" * 80)
    
    # Initialize
    classifier = FinalModelClassifier()
    
    # Test article
    title = "Deep learning for drug discovery and development"
    abstract = "Machine learning methods are increasingly being used in pharmaceutical research to identify potential drug candidates and predict their efficacy."
    
    print("\nTest Article:")
    print(f"Title: {title}")
    print(f"Abstract: {abstract[:100]}...")
    
    # Test direct classification
    print("\n1. Direct Classification (Top 5):")
    print("-" * 80)
    result = classifier.classify(title, abstract)
    print(f"Predicted: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # Test with candidate paths
    print("\n2. Re-ranking with Candidate Paths:")
    print("-" * 80)
    candidate_paths = [
        "Drug Discovery",
        "Machine Learning",
        "Bioinformatics",
        "Chemical Engineering",
        "Natural Language Processing"
    ]
    
    result = classifier.classify_with_paths(
        title=title,
        abstract=abstract,
        candidate_paths=candidate_paths
    )
    
    print(f"Best Path: {result['path']}")
    print(f"Model Score: {result.get('model_score', 0):.4f}")
    print(f"\nAll Candidate Scores:")
    for item in result.get('all_scores', []):
        print(f"  {item['path']}: {item['model_score']:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Testing complete!")
