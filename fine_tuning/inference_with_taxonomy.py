"""
Enhanced Inference with Taxonomy Validation
This version ensures the model only outputs valid taxonomy paths
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict, Optional, Set
import argparse
from tqdm import tqdm
from difflib import get_close_matches


class TaxonomyConstrainedClassifier:
    """Classifier with taxonomy validation."""
    
    def __init__(
        self,
        model_path: str,
        taxonomy_path: str,
        base_model: Optional[str] = None
    ):
        """
        Initialize the classifier with taxonomy constraints.
        
        Args:
            model_path: Path to fine-tuned model
            taxonomy_path: Path to taxonomy_reference.json
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
        
        # Load model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        
        if os.path.exists(adapter_config_path) and base_model:
            print(f"Loading base model {base_model} and applying LoRA adapter...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            print("Loading full fine-tuned model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        
        # Load taxonomy
        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            taxonomy_ref = json.load(f)
        
        self.valid_paths = set(taxonomy_ref['valid_paths'])
        self.domains = taxonomy_ref.get('domains', {})
        
        print(f"Loaded {len(self.valid_paths)} valid classification paths")
        print(f"Domains: {', '.join(self.domains.keys())}")
        print("Model ready!")
    
    def validate_and_correct_path(self, predicted_path: str) -> tuple[str, bool]:
        """
        Validate predicted path and suggest correction if invalid.
        
        Args:
            predicted_path: Path predicted by model
            
        Returns:
            Tuple of (corrected_path, is_valid)
        """
        # Clean the prediction
        predicted_path = predicted_path.strip()
        
        # Check if valid
        if predicted_path in self.valid_paths:
            return predicted_path, True
        
        # Try case-insensitive match
        for valid_path in self.valid_paths:
            if valid_path.lower() == predicted_path.lower():
                return valid_path, True
        
        # Find closest matches
        close_matches = get_close_matches(
            predicted_path,
            self.valid_paths,
            n=1,
            cutoff=0.6
        )
        
        if close_matches:
            print(f"⚠️  Invalid path corrected:")
            print(f"   Predicted: {predicted_path}")
            print(f"   Corrected: {close_matches[0]}")
            return close_matches[0], False
        
        # If no close match, return most common path in same domain
        parts = predicted_path.split(' > ')
        if parts:
            domain = parts[0]
            if domain in self.domains and self.domains[domain]:
                fallback = self.domains[domain][0]
                print(f"⚠️  Invalid path, using domain fallback:")
                print(f"   Predicted: {predicted_path}")
                print(f"   Fallback: {fallback}")
                return fallback, False
        
        # Last resort: use first valid path
        fallback = list(self.valid_paths)[0]
        print(f"❌ Could not correct invalid path: {predicted_path}")
        print(f"   Using fallback: {fallback}")
        return fallback, False
    
    def classify(
        self,
        title: str,
        abstract: str,
        max_new_tokens: int = 100,
        temperature: float = 0.1,
        top_p: float = 0.9,
        validate: bool = True
    ) -> Dict[str, any]:
        """
        Classify a single article with taxonomy validation.
        
        Args:
            title: Article title
            abstract: Article abstract
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            validate: Whether to validate against taxonomy
            
        Returns:
            Dictionary with classification and validation info
        """
        # Format prompt
        instruction = """Classify the following research article into its appropriate taxonomy path based on the title and abstract. Choose from the predefined research taxonomy covering: Natural Science, Engineering and Technology, Medical and Health Science, Agricultural Science, Social Science, Humanity and Art, and Interdisciplinary Fields. Provide the full hierarchical path (e.g., "Domain > Field > Subfield > Specialty")."""
        
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
        
        # Extract classification
        if "Classification:" in generated_text:
            raw_classification = generated_text.split("Classification:")[-1].strip()
        else:
            raw_classification = generated_text.strip()
        
        # Remove any trailing text after the path
        raw_classification = raw_classification.split('\n')[0].strip()
        
        # Validate and correct
        if validate:
            final_classification, is_valid = self.validate_and_correct_path(raw_classification)
        else:
            final_classification = raw_classification
            is_valid = raw_classification in self.valid_paths
        
        return {
            'classification': final_classification,
            'raw_prediction': raw_classification,
            'is_valid': is_valid,
            'corrected': raw_classification != final_classification
        }
    
    def classify_batch(
        self,
        articles: List[Dict[str, str]],
        batch_size: int = 4,
        **kwargs
    ) -> List[Dict]:
        """
        Classify multiple articles.
        
        Args:
            articles: List of dicts with 'title' and 'abstract'
            batch_size: Number of articles to process at once
            **kwargs: Additional arguments for classify()
            
        Returns:
            List of classification results
        """
        results = []
        
        for i in tqdm(range(0, len(articles), batch_size), desc="Classifying"):
            batch = articles[i:i + batch_size]
            
            for article in batch:
                result = self.classify(
                    article['title'],
                    article['abstract'],
                    **kwargs
                )
                results.append(result)
        
        return results


def evaluate_model(
    model_path: str,
    taxonomy_path: str,
    test_data_path: str,
    base_model: Optional[str] = None,
    output_file: Optional[str] = None
):
    """Evaluate model with taxonomy validation."""
    print("\n" + "="*60)
    print("MODEL EVALUATION WITH TAXONOMY VALIDATION")
    print("="*60 + "\n")
    
    # Load classifier
    classifier = TaxonomyConstrainedClassifier(model_path, taxonomy_path, base_model)
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Test samples: {len(test_data)}")
    
    # Prepare articles
    articles = [
        {
            'title': item['input'].split('Abstract:')[0].replace('Title:', '').strip(),
            'abstract': item['input'].split('Abstract:')[1].strip() if 'Abstract:' in item['input'] else '',
            'true_label': item['output']
        }
        for item in test_data
    ]
    
    # Get predictions
    print("\nGenerating predictions with taxonomy validation...")
    results = classifier.classify_batch(articles)
    
    # Compute metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60 + "\n")
    
    true_labels = [a['true_label'] for a in articles]
    predictions = [r['classification'] for r in results]
    
    # Validation statistics
    total_valid = sum(r['is_valid'] for r in results)
    total_corrected = sum(r['corrected'] for r in results)
    
    print(f"Taxonomy Validation:")
    print(f"  Valid predictions: {total_valid}/{len(results)} ({100*total_valid/len(results):.2f}%)")
    print(f"  Auto-corrected: {total_corrected} ({100*total_corrected/len(results):.2f}%)")
    
    # Exact match accuracy
    exact_matches = sum(p == t for p, t in zip(predictions, true_labels))
    exact_accuracy = exact_matches / len(predictions)
    
    print(f"\nExact Match Accuracy: {exact_accuracy:.4f} ({exact_matches}/{len(predictions)})")
    
    # Hierarchical accuracy
    def hierarchical_accuracy(predictions, true_labels, level):
        pred_levels = [' > '.join(p.split(' > ')[:level]) if ' > ' in p else p for p in predictions]
        true_levels = [' > '.join(t.split(' > ')[:level]) if ' > ' in t else t for t in true_labels]
        matches = sum(p == t for p, t in zip(pred_levels, true_levels))
        return matches / len(predictions)
    
    print("\nHierarchical Accuracy:")
    for level in range(1, 6):
        acc = hierarchical_accuracy(predictions, true_labels, level)
        print(f"  Level {level}: {acc:.4f}")
    
    # Save detailed results
    if output_file:
        detailed_results = []
        for article, result, pred in zip(articles, results, predictions):
            detailed_results.append({
                'title': article['title'],
                'abstract': article['abstract'][:200] + '...',
                'true_classification': article['true_label'],
                'predicted_classification': pred,
                'raw_prediction': result['raw_prediction'],
                'is_valid': result['is_valid'],
                'was_corrected': result['corrected'],
                'correct': pred == article['true_label']
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")
    
    # Show examples
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    for i in range(min(5, len(articles))):
        print(f"\nExample {i+1}:")
        print(f"Title: {articles[i]['title'][:80]}...")
        print(f"True: {articles[i]['true_label']}")
        print(f"Pred: {predictions[i]}")
        if results[i]['corrected']:
            print(f"  (Auto-corrected from: {results[i]['raw_prediction']})")
        print(f"Match: {'✓' if predictions[i] == articles[i]['true_label'] else '✗'}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Taxonomy-constrained inference and evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--taxonomy_path", type=str, default="./processed_data/taxonomy_reference.json", help="Path to taxonomy reference")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name (for LoRA)")
    parser.add_argument("--mode", type=str, choices=["evaluate"], default="evaluate", help="Operation mode")
    parser.add_argument("--test_data", type=str, default="./processed_data/test_instruction.json", help="Test data path")
    parser.add_argument("--output_file", type=str, default="./evaluation_results_validated.json", help="Output file")
    
    args = parser.parse_args()
    
    if args.mode == "evaluate":
        evaluate_model(
            args.model_path,
            args.taxonomy_path,
            args.test_data,
            args.base_model,
            args.output_file
        )


if __name__ == "__main__":
    main()
