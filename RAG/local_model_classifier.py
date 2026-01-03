"""
Local Model Classifier - Uses fine-tuned SciBERT model with LoRA
Replaces API-based LLM classification with local model inference
"""

import os
import torch
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)


class LocalModelClassifier:
    """Fine-tuned SciBERT classifier using LoRA adapters"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize local model classifier
        
        Args:
            model_path: Path to the fine-tuned model with LoRA adapters
            device: Device to run inference on (cuda/cpu)
        """
        # Default model path
        if model_path is None:
            model_path = str(Path(__file__).parent / "best_models" / "scibert_lora_final")
        
        self.model_path = model_path
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load PEFT config to get base model
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = peft_config.base_model_name_or_path
        
        logger.info(f"Base model: {base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # The model was trained with 862 labels (not 1449)
        # Load base model with correct number of labels from training
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=862,  # Number from your fine-tuning
            problem_type="single_label_classification"
        )
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Model metadata
        self.model_name = f"SciBERT-LoRA ({base_model_name})"
        self.num_labels = 862  # The actual number of classes the model was trained on
        
        logger.info(f"Model loaded successfully with {self.num_labels} labels")
    
    def classify_with_paths(
        self,
        title: str,
        abstract: str,
        candidate_paths: List[str],
        candidate_similarities: Optional[List[float]] = None,
        return_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Re-rank candidate paths using the fine-tuned model
        
        Strategy: Since we don't have the index->path mapping, we use the model
        to generate semantic embeddings and re-score the candidates.
        
        Args:
            title: Article title
            abstract: Article abstract
            candidate_paths: List of candidate taxonomy paths from retrieval
            candidate_similarities: Original similarity scores from retrieval
            return_scores: Whether to return confidence scores for all candidates
            
        Returns:
            Dictionary with selected path and confidence
        """
        if not candidate_paths:
            return {
                'predicted_index': -1,
                'confidence': 0.0,
                'path': None,
                'valid': False
            }
        
        # Combine title and abstract
        article_text = f"{title} [SEP] {abstract}"
        
        # Truncate if too long
        max_length = 512
        if len(article_text) > max_length * 4:
            article_text = article_text[:max_length * 4]
        
        # Re-rank candidates by scoring each with the model
        candidate_scores = []
        
        for i, path in enumerate(candidate_paths):
            # Create a combined text: article + path for scoring
            # This helps the model understand if this path is relevant
            combined_text = f"{article_text} [SEP] Path: {path}"
            
            # Tokenize
            inputs = self.tokenizer(
                combined_text,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
                # Use max probability as relevance score for this path
                max_prob = torch.max(probabilities).item()
                
                # Combine with retrieval similarity if available
                if candidate_similarities and i < len(candidate_similarities):
                    # Weighted combination: 60% model, 40% retrieval
                    combined_score = 0.6 * max_prob + 0.4 * candidate_similarities[i]
                else:
                    combined_score = max_prob
                
                candidate_scores.append({
                    'path': path,
                    'model_score': max_prob,
                    'retrieval_score': candidate_similarities[i] if candidate_similarities and i < len(candidate_similarities) else 0.0,
                    'combined_score': combined_score
                })
        
        # Sort by combined score
        candidate_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top scored path
        best_candidate = candidate_scores[0]
        
        result = {
            'predicted_index': 0,  # Re-ranked index
            'confidence': best_candidate['combined_score'],
            'model_score': best_candidate['model_score'],
            'retrieval_score': best_candidate['retrieval_score'],
            'path': best_candidate['path'],
            'valid': True,
            'reranked': True
        }
        
        if return_scores:
            result['all_candidates'] = candidate_scores
        
        return result
    
    def classify_article(
        self,
        title: str,
        abstract: str,
        relevant_paths: List[str],
        retrieval_similarities: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete classification compatible with LLM classifier interface
        
        Uses the model to re-rank the top retrieved paths
        
        Args:
            title: Article title
            abstract: Article abstract
            relevant_paths: Retrieved taxonomy paths
            retrieval_similarities: Similarity scores from retrieval
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
            title, 
            abstract, 
            relevant_paths,
            candidate_similarities=retrieval_similarities,
            return_scores=True
        )
        
        # Format response to match LLM classifier interface
        confidence_label = "High" if result['confidence'] > 0.7 else \
                          "Medium" if result['confidence'] > 0.5 else "Low"
        
        reasoning = f"Re-ranked {len(relevant_paths)} candidates using fine-tuned SciBERT. " \
                   f"Model score: {result.get('model_score', 0):.3f}, " \
                   f"Retrieval score: {result.get('retrieval_score', 0):.3f}"
        
        return {
            'classification': {
                'path': result['path'],
                'confidence': confidence_label,
                'confidence_score': result['confidence'],
                'model_score': result.get('model_score', 0),
                'retrieval_score': result.get('retrieval_score', 0),
                'reasoning': reasoning,
                'raw_response': f"Re-ranking with combined score: {result['confidence']:.4f}"
            },
            'prompt_length': 0,  # Not applicable for local model
            'response_length': 0,
        }
    
    def parse_classification_response(self, response: str) -> Dict[str, Optional[str]]:
        """
        Compatibility method - not used for local model
        """
        return {
            'path': None,
            'confidence': None,
            'reasoning': None,
            'raw_response': response
        }
    
    def create_classification_prompt(
        self,
        title: str,
        abstract: str,
        relevant_paths: List[str],
        include_reasoning: bool = False
    ) -> str:
        """
        Compatibility method - returns formatted text for logging
        """
        return f"Title: {title}\n\nAbstract: {abstract[:200]}..."
