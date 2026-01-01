"""
Inference and Evaluation Script for Research Article Classification
Use this script to classify articles or evaluate model performance
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


class ArticleClassifier:
    """Classifier for research articles."""
    
    def __init__(self, model_path: str, base_model: Optional[str] = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to fine-tuned model
            base_model: Base model name (if using LoRA adapter)
        """
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if this is a LoRA adapter or full model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        
        if os.path.exists(adapter_config_path) and base_model:
            # Load base model and apply adapter
            print(f"Loading base model {base_model} and applying LoRA adapter...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            # Load full model
            print("Loading full fine-tuned model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def classify(
        self,
        title: str,
        abstract: str,
        max_new_tokens: int = 100,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> str:
        """
        Classify a single article.
        
        Args:
            title: Article title
            abstract: Article abstract
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Classification path
        """
        # Format prompt
        instruction = "Classify the following research article into its appropriate taxonomy path based on the title and abstract. Provide the full classification path from field to subfield."
        input_text = f"Title: {title}\n\nAbstract: {abstract}"
        prompt = f"{instruction}\n\n{input_text}\n\nClassification:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract classification (after "Classification:")
        if "Classification:" in generated_text:
            classification = generated_text.split("Classification:")[-1].strip()
        else:
            classification = generated_text.strip()
        
        return classification
    
    def classify_batch(
        self,
        articles: List[Dict[str, str]],
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Classify multiple articles.
        
        Args:
            articles: List of dicts with 'title' and 'abstract'
            batch_size: Number of articles to process at once
            **kwargs: Additional arguments for classify()
            
        Returns:
            List of classifications
        """
        predictions = []
        
        for i in tqdm(range(0, len(articles), batch_size), desc="Classifying"):
            batch = articles[i:i + batch_size]
            
            for article in batch:
                pred = self.classify(
                    article['title'],
                    article['abstract'],
                    **kwargs
                )
                predictions.append(pred)
        
        return predictions


def evaluate_model(
    model_path: str,
    test_data_path: str,
    base_model: Optional[str] = None,
    output_file: Optional[str] = None
):
    """
    Evaluate model on test set.
    
    Args:
        model_path: Path to fine-tuned model
        test_data_path: Path to test data JSON
        base_model: Base model name (if using LoRA)
        output_file: Path to save detailed results
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60 + "\n")
    
    # Load classifier
    classifier = ArticleClassifier(model_path, base_model)
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Test samples: {len(test_data)}")
    
    # Prepare articles
    articles = [
        {
            'title': item['input'].split('Abstract:')[0].replace('Title:', '').strip(),
            'abstract': item['input'].split('Abstract:')[1].strip(),
            'true_label': item['output']
        }
        for item in test_data
    ]
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = classifier.classify_batch(articles)
    
    # Compute metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60 + "\n")
    
    true_labels = [a['true_label'] for a in articles]
    
    # Exact match accuracy
    exact_matches = sum(p == t for p, t in zip(predictions, true_labels))
    exact_accuracy = exact_matches / len(predictions)
    
    print(f"Exact Match Accuracy: {exact_accuracy:.4f} ({exact_matches}/{len(predictions)})")
    
    # Hierarchical accuracy (matching at different levels)
    def hierarchical_accuracy(predictions, true_labels, level):
        """Calculate accuracy at specific taxonomy level."""
        pred_levels = [' > '.join(p.split(' > ')[:level]) if ' > ' in p else p for p in predictions]
        true_levels = [' > '.join(t.split(' > ')[:level]) if ' > ' in t else t for t in true_labels]
        matches = sum(p == t for p, t in zip(pred_levels, true_levels))
        return matches / len(predictions)
    
    print("\nHierarchical Accuracy:")
    for level in range(1, 5):
        acc = hierarchical_accuracy(predictions, true_labels, level)
        print(f"  Level {level}: {acc:.4f}")
    
    # Save detailed results
    if output_file:
        results = []
        for article, pred in zip(articles, predictions):
            results.append({
                'title': article['title'],
                'abstract': article['abstract'][:200] + '...',  # Truncate for readability
                'true_classification': article['true_label'],
                'predicted_classification': pred,
                'correct': pred == article['true_label']
            })
        
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Also save as CSV for easier viewing
        df = pd.DataFrame(results)
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"\nDetailed results saved to:")
        print(f"  - {output_file}")
        print(f"  - {csv_file}")
    
    # Print some examples
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    for i in range(min(3, len(articles))):
        print(f"\nExample {i+1}:")
        print(f"Title: {articles[i]['title'][:80]}...")
        print(f"True: {articles[i]['true_label']}")
        print(f"Pred: {predictions[i]}")
        print(f"Match: {'✓' if predictions[i] == articles[i]['true_label'] else '✗'}")


def interactive_mode(model_path: str, base_model: Optional[str] = None):
    """
    Interactive mode for classifying articles.
    
    Args:
        model_path: Path to fine-tuned model
        base_model: Base model name (if using LoRA)
    """
    print("\n" + "="*60)
    print("INTERACTIVE CLASSIFICATION MODE")
    print("="*60 + "\n")
    
    classifier = ArticleClassifier(model_path, base_model)
    
    print("Enter article details to classify (or 'quit' to exit)\n")
    
    while True:
        print("-" * 60)
        title = input("Title: ").strip()
        
        if title.lower() in ['quit', 'exit', 'q']:
            break
        
        abstract = input("Abstract: ").strip()
        
        if not title or not abstract:
            print("Both title and abstract are required!")
            continue
        
        print("\nClassifying...")
        classification = classifier.classify(title, abstract)
        
        print(f"\nClassification: {classification}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Inference and evaluation for research article classification")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (required if using LoRA adapter)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["evaluate", "interactive", "classify"],
        default="interactive",
        help="Operation mode"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./processed_data/test_instruction.json",
        help="Path to test data (for evaluate mode)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./evaluation_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Article title (for classify mode)"
    )
    parser.add_argument(
        "--abstract",
        type=str,
        default=None,
        help="Article abstract (for classify mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "evaluate":
        evaluate_model(
            args.model_path,
            args.test_data,
            args.base_model,
            args.output_file
        )
    elif args.mode == "interactive":
        interactive_mode(args.model_path, args.base_model)
    elif args.mode == "classify":
        if not args.title or not args.abstract:
            print("Error: --title and --abstract are required in classify mode")
            return
        
        classifier = ArticleClassifier(args.model_path, args.base_model)
        classification = classifier.classify(args.title, args.abstract)
        print(f"\nClassification: {classification}")


if __name__ == "__main__":
    main()
