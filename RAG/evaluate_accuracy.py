"""
RAG Accuracy Evaluation - Test with 20 Labeled Articles
Calculates exact match accuracy, domain accuracy, and hierarchical metrics
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

from rag_pipeline import RAGClassificationPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGAccuracyEvaluator:
    """Evaluate RAG classification accuracy against labeled data"""
    
    def __init__(self, test_data_path: str = "test_data_20_articles.json"):
        """Initialize evaluator with test data"""
        self.test_data_path = Path(test_data_path)
        self.test_data = self._load_test_data()
        self.results = []
        
    def _load_test_data(self) -> List[Dict]:
        """Load labeled test articles"""
        if not self.test_data_path.exists():
            raise FileNotFoundError(
                f"Test data file not found: {self.test_data_path}\n"
                f"Please create a JSON file with format:\n"
                f"[\n"
                f'  {{"title": "...", "abstract": "...", "ground_truth": "Domain > Field > ..."}},\n'
                f"  ...\n"
                f"]"
            )
        
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} test articles from {self.test_data_path}")
        return data
    
    def run_evaluation(self, pipeline: RAGClassificationPipeline) -> Dict[str, Any]:
        """
        Run complete evaluation
        
        Returns:
            Dictionary with accuracy metrics and detailed results
        """
        print("\n" + "="*70)
        print("RAG CLASSIFICATION ACCURACY EVALUATION")
        print("="*70)
        print(f"Test Dataset: {self.test_data_path}")
        print(f"Total Articles: {len(self.test_data)}")
        print("="*70)
        
        self.results = []
        successful = 0
        failed = 0
        
        # Classify each article
        for i, article in enumerate(self.test_data, 1):
            print(f"\n[{i}/{len(self.test_data)}] Processing: {article['title'][:60]}...")
            
            try:
                start_time = time.time()
                
                result = pipeline.classify_article(
                    title=article['title'],
                    abstract=article['abstract'],
                    return_metadata=True
                )
                
                inference_time = time.time() - start_time
                
                # Extract classification
                classified_path = result['classification']['path']
                confidence = result['classification']['confidence']
                ground_truth = article['ground_truth']
                
                # Compare
                exact_match = (classified_path == ground_truth)
                
                # Store result
                self.results.append({
                    'article_id': i,
                    'title': article['title'],
                    'ground_truth': ground_truth,
                    'predicted': classified_path,
                    'confidence': confidence,
                    'exact_match': exact_match,
                    'inference_time': inference_time,
                    'retrieved_paths': result.get('metadata', {}).get('retrieved_paths', []),
                    'retrieval_scores': result.get('metadata', {}).get('retrieval_scores', [])
                })
                
                successful += 1
                
                # Show result
                match_symbol = "✓" if exact_match else "✗"
                print(f"  {match_symbol} Predicted: {classified_path}")
                print(f"    Ground Truth: {ground_truth}")
                print(f"    Confidence: {confidence} | Time: {inference_time:.2f}s")
                
            except Exception as e:
                failed += 1
                logger.error(f"  ✗ Classification failed: {str(e)[:100]}")
                self.results.append({
                    'article_id': i,
                    'title': article['title'],
                    'ground_truth': article['ground_truth'],
                    'predicted': None,
                    'confidence': None,
                    'exact_match': False,
                    'error': str(e)
                })
        
        # Calculate metrics
        print("\n" + "="*70)
        print("CALCULATING METRICS...")
        print("="*70)
        
        metrics = self._calculate_metrics()
        
        # Display results
        self._display_results(metrics, successful, failed)
        
        # Save results
        self._save_results(metrics)
        
        return {
            'metrics': metrics,
            'results': self.results,
            'summary': {
                'total': len(self.test_data),
                'successful': successful,
                'failed': failed
            }
        }
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate accuracy metrics"""
        valid_results = [r for r in self.results if r['predicted'] is not None]
        
        if not valid_results:
            return {
                'exact_match_accuracy': 0.0,
                'domain_accuracy': 0.0,
                'field_accuracy': 0.0,
                'subfield_accuracy': 0.0,
                'avg_inference_time': 0.0,
                'total_evaluated': 0
            }
        
        # Exact match accuracy
        exact_matches = sum(1 for r in valid_results if r['exact_match'])
        exact_match_acc = exact_matches / len(valid_results) * 100
        
        # Hierarchical accuracy (domain, field, subfield)
        domain_matches = 0
        field_matches = 0
        subfield_matches = 0
        
        for r in valid_results:
            pred_parts = r['predicted'].split(' > ')
            true_parts = r['ground_truth'].split(' > ')
            
            # Domain (level 1)
            if len(pred_parts) >= 1 and len(true_parts) >= 1:
                if pred_parts[0] == true_parts[0]:
                    domain_matches += 1
            
            # Field (level 2)
            if len(pred_parts) >= 2 and len(true_parts) >= 2:
                if pred_parts[0] == true_parts[0] and pred_parts[1] == true_parts[1]:
                    field_matches += 1
            
            # Subfield (level 3)
            if len(pred_parts) >= 3 and len(true_parts) >= 3:
                if (pred_parts[0] == true_parts[0] and 
                    pred_parts[1] == true_parts[1] and 
                    pred_parts[2] == true_parts[2]):
                    subfield_matches += 1
        
        domain_acc = domain_matches / len(valid_results) * 100
        field_acc = field_matches / len(valid_results) * 100
        subfield_acc = subfield_matches / len(valid_results) * 100
        
        # Average inference time
        avg_time = sum(r.get('inference_time', 0) for r in valid_results) / len(valid_results)
        
        # Confidence distribution
        high_conf = sum(1 for r in valid_results if r.get('confidence') == 'High')
        medium_conf = sum(1 for r in valid_results if r.get('confidence') == 'Medium')
        low_conf = sum(1 for r in valid_results if r.get('confidence') == 'Low')
        
        return {
            'exact_match_accuracy': exact_match_acc,
            'domain_accuracy': domain_acc,
            'field_accuracy': field_acc,
            'subfield_accuracy': subfield_acc,
            'hierarchical_accuracies': {
                'level_1': domain_acc,
                'level_2': field_acc,
                'level_3': subfield_acc
            },
            'exact_matches': exact_matches,
            'total_evaluated': len(valid_results),
            'avg_inference_time': avg_time,
            'confidence_distribution': {
                'high': high_conf,
                'medium': medium_conf,
                'low': low_conf
            }
        }
    
    def _display_results(self, metrics: Dict, successful: int, failed: int):
        """Display formatted results"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        print(f"\n📊 CLASSIFICATION SUMMARY:")
        print(f"  Total Articles: {len(self.test_data)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        print(f"\n🎯 ACCURACY METRICS:")
        print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}% ({metrics['exact_matches']}/{metrics['total_evaluated']})")
        print(f"  Domain Accuracy (Level 1): {metrics['domain_accuracy']:.2f}%")
        print(f"  Field Accuracy (Level 2): {metrics['field_accuracy']:.2f}%")
        print(f"  Subfield Accuracy (Level 3): {metrics['subfield_accuracy']:.2f}%")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"  Average Inference Time: {metrics['avg_inference_time']:.2f}s")
        print(f"  Total Time: {metrics['avg_inference_time'] * successful:.2f}s")
        
        print(f"\n💪 CONFIDENCE DISTRIBUTION:")
        conf_dist = metrics['confidence_distribution']
        print(f"  High: {conf_dist['high']} ({conf_dist['high']/metrics['total_evaluated']*100:.1f}%)")
        print(f"  Medium: {conf_dist['medium']} ({conf_dist['medium']/metrics['total_evaluated']*100:.1f}%)")
        print(f"  Low: {conf_dist['low']} ({conf_dist['low']/metrics['total_evaluated']*100:.1f}%)")
        
        # Show misclassifications
        misclassified = [r for r in self.results if r['predicted'] and not r['exact_match']]
        if misclassified:
            print(f"\n❌ MISCLASSIFICATIONS ({len(misclassified)}):")
            for r in misclassified[:5]:  # Show first 5
                print(f"\n  Article: {r['title'][:60]}...")
                print(f"  Ground Truth: {r['ground_truth']}")
                print(f"  Predicted:    {r['predicted']}")
                print(f"  Confidence:   {r['confidence']}")
            
            if len(misclassified) > 5:
                print(f"\n  ... and {len(misclassified) - 5} more")
        
        print("\n" + "="*70)
    
    def _save_results(self, metrics: Dict):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = Path(f"results/accuracy_test_{timestamp}.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'test_data_file': str(self.test_data_path),
                'metrics': metrics,
                'detailed_results': self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Save CSV summary
        csv_file = Path(f"results/accuracy_test_{timestamp}.csv")
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("article_id,title,ground_truth,predicted,confidence,exact_match,inference_time\n")
            for r in self.results:
                if r['predicted']:
                    f.write(f"{r['article_id']},\"{r['title']}\",\"{r['ground_truth']}\",\"{r['predicted']}\",{r['confidence']},{r['exact_match']},{r.get('inference_time', 0):.2f}\n")
        
        print(f"💾 CSV saved to: {csv_file}")


def main():
    """Main evaluation function"""
    
    # Check if test data exists
    test_data_file = "test_data_20_articles.json"
    if not Path(test_data_file).exists():
        print("\n" + "="*70)
        print("⚠️  TEST DATA FILE NOT FOUND")
        print("="*70)
        print(f"\nPlease create '{test_data_file}' with your 20 labeled articles.")
        print("\nExpected format:")
        print("""
[
  {
    "title": "Your Article Title",
    "abstract": "Article abstract text...",
    "ground_truth": "Domain > Field > Subfield > Specialty > Topic"
  },
  {
    "title": "Another Article",
    "abstract": "Abstract...",
    "ground_truth": "Domain > Field > ..."
  }
]
        """)
        print("\nExample ground_truth paths:")
        print("  • Natural Science > Computer and Information Science > Artificial Intelligence > Machine Learning > Deep Learning")
        print("  • Natural Science > Environmental Science > Climate Science > Climate Change > Climate Impact")
        print("  • Social Sciences > Economics > Macroeconomics > Economic Growth > Development Economics")
        print("\n" + "="*70)
        return
    
    # Initialize pipeline
    print("\n" + "="*70)
    print("INITIALIZING RAG PIPELINE")
    print("="*70)
    
    try:
        pipeline = RAGClassificationPipeline(auto_setup=True)
        print(f"✓ Pipeline initialized successfully")
        print(f"  Database paths: {pipeline.db_manager.collection.count()}")
        
        # Check API key status
        if pipeline.llm_classifier:
            status = pipeline.llm_classifier.get_api_key_status()
            print(f"  API keys loaded: {status['total_keys']}")
            print(f"  Healthy keys: {status['healthy_keys']}")
    
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return
    
    # Run evaluation
    evaluator = RAGAccuracyEvaluator(test_data_file)
    
    try:
        evaluation_results = evaluator.run_evaluation(pipeline)
        
        print("\n✓ Evaluation complete!")
        print(f"\nFinal Accuracy: {evaluation_results['metrics']['exact_match_accuracy']:.2f}%")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
