"""
LLM Classifier - Interface for Google Gemini via AI Studio API
"""

import os
import google.generativeai as genai
from typing import Dict, Any, Optional, List
import logging
import re
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LLMClassifier:
    """Google Gemini classifier using AI Studio API with automatic key rotation"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        model_name: str = "gemini-2.0-flash",
        api_base_url: Optional[str] = None
    ):
        """
        Initialize LLM classifier with Google AI Studio API
        
        Args:
            api_key: Single Google API key (legacy support)
            api_keys: List of Google API keys for automatic rotation
            model_name: Model name (gemini-2.0-flash-exp, gemini-2.5-pro, gemini-1.5-flash, gemini-1.5-pro)
            api_base_url: Not used (kept for compatibility)
        """
        # Load API keys from environment or parameters
        if api_keys:
            self.api_keys = api_keys
        elif os.getenv('GOOGLE_API_KEYS'):
            # Multiple keys separated by comma
            self.api_keys = [k.strip() for k in os.getenv('GOOGLE_API_KEYS').split(',') if k.strip()]
        elif api_key:
            self.api_keys = [api_key]
        elif os.getenv('GOOGLE_API_KEY'):
            self.api_keys = [os.getenv('GOOGLE_API_KEY')]
        else:
            raise ValueError(
                "API key required. Set GOOGLE_API_KEYS (comma-separated) or GOOGLE_API_KEY environment variable."
            )
        
        self.model_name = model_name
        self.current_key_index = 0
        self.key_failure_count = {i: 0 for i in range(len(self.api_keys))}
        
        # Configure with first API key
        self._configure_api_key(self.api_keys[0])
        
        logger.info(f"Initialized LLM classifier with {len(self.api_keys)} API key(s) and model: {model_name}")
    
    def _configure_api_key(self, api_key: str):
        """Configure Google Generative AI with specific API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.current_api_key = api_key
    
    def _rotate_api_key(self) -> bool:
        """Rotate to next available API key"""
        if len(self.api_keys) == 1:
            logger.warning("Only one API key available, cannot rotate")
            return False
        
        original_index = self.current_key_index
        attempts = 0
        
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            # Skip keys that have failed too many times (more than 5 consecutive failures)
            if self.key_failure_count[self.current_key_index] < 5:
                self._configure_api_key(self.api_keys[self.current_key_index])
                logger.info(f"Rotated to API key #{self.current_key_index + 1}")
                return True
            
            attempts += 1
        
        logger.error("All API keys have failed multiple times")
        return False
    
    def create_classification_prompt(
        self,
        title: str,
        abstract: str,
        relevant_paths: List[str],
        include_reasoning: bool = False  # Disabled by default for speed
    ) -> str:
        """
        Create optimized prompt for classification (shortened for speed)
        
        Args:
            title: Article title
            abstract: Article abstract
            relevant_paths: List of retrieved taxonomy paths
            include_reasoning: Include reasoning in response
            
        Returns:
            Formatted prompt string
        """
        # Format paths with numbering
        paths_text = "\n".join([
            f"{i+1}. {path}"
            for i, path in enumerate(relevant_paths)
        ])
        
        # Truncate abstract if too long (save tokens)
        max_abstract_len = 500
        if len(abstract) > max_abstract_len:
            abstract = abstract[:max_abstract_len] + "..."
        
        # Shortened prompt for faster processing
        prompt = f"""Classify this research article into ONE taxonomy path from the list.

Title: {title}
Abstract: {abstract}

Paths:
{paths_text}

Reply with ONLY:
Path: [exact path from list]
Confidence: [High/Medium/Low]"""
        
        if include_reasoning:
            prompt += "\nReasoning: [1 sentence]"
        
        return prompt
    
    def classify(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 256,
        top_p: float = 0.9,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Generate classification using Google AI Studio API with automatic key rotation
        
        Args:
            prompt: Classification prompt
            temperature: Sampling temperature (lower = more deterministic, 0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0-1)
            max_retries: Maximum retry attempts with key rotation
            **kwargs: Additional API parameters
            
        Returns:
            LLM response text
        """
        # Build system message + user prompt
        full_prompt = "You are an expert research article classifier with deep knowledge across all scientific domains.\n\n" + prompt
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Configure generation parameters
                generation_config = genai.GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_tokens,
                )
                
                # Generate response
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                # Success - reset failure count for this key
                self.key_failure_count[self.current_key_index] = 0
                
                # Extract text
                return response.text.strip()
                    
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Increment failure count
                self.key_failure_count[self.current_key_index] += 1
                
                # Check if it's a rate limit or quota error
                is_rate_limit = any(keyword in error_msg for keyword in 
                    ['rate limit', 'quota', 'resource exhausted', '429', 'quota exceeded'])
                
                if is_rate_limit:
                    logger.warning(f"API key #{self.current_key_index + 1} hit rate limit: {e}")
                    
                    # Try to rotate to next key
                    if attempt < max_retries - 1 and self._rotate_api_key():
                        logger.info(f"Retrying with new API key (attempt {attempt + 2}/{max_retries})")
                        time.sleep(0.5)  # Brief pause before retry
                        continue
                    else:
                        logger.error("No more API keys available or all keys exhausted")
                        break
                else:
                    # Other errors - log and retry with same key
                    logger.error(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Brief pause before retry
                    else:
                        break
        
        # All retries failed
        logger.error(f"All {max_retries} attempts failed. Last error: {last_exception}")
        raise last_exception
    
    def parse_classification_response(
        self,
        response: str
    ) -> Dict[str, Optional[str]]:
        """
        Parse structured response from LLM
        
        Args:
            response: LLM response text
            
        Returns:
            Dictionary with parsed fields
        """
        # Extract path
        path_match = re.search(r'Path:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        path = path_match.group(1).strip() if path_match else None
        
        # Clean up common formatting
        if path:
            path = path.strip('[]"\'')
            # Remove any trailing punctuation or notes
            path = re.split(r'\s*[\(\[]', path)[0].strip()
        
        # Extract confidence
        confidence_match = re.search(
            r'Confidence:\s*(High|Medium|Low)',
            response,
            re.IGNORECASE
        )
        confidence = confidence_match.group(1).capitalize() if confidence_match else None
        
        # Extract reasoning
        reasoning_match = re.search(
            r'Reasoning:\s*(.+?)(?:\n\n|\Z)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        
        return {
            'path': path,
            'confidence': confidence,
            'reasoning': reasoning,
            'raw_response': response
        }
    
    def classify_article(
        self,
        title: str,
        abstract: str,
        relevant_paths: List[str],
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Complete classification pipeline
        
        Args:
            title: Article title
            abstract: Article abstract
            relevant_paths: Retrieved taxonomy paths
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Classification result with metadata
        """
        # Create prompt
        prompt = self.create_classification_prompt(
            title,
            abstract,
            relevant_paths
        )
        
        # Generate response
        response = self.classify(prompt, **generation_kwargs)
        
        # Parse response
        parsed = self.parse_classification_response(response)
        
        return {
            'classification': parsed,
            'prompt_length': len(prompt),
            'response_length': len(response),
        }
    
    def get_api_key_status(self) -> Dict[str, Any]:
        """
        Get current status of all API keys
        
        Returns:
            Dictionary with API key usage statistics
        """
        return {
            'total_keys': len(self.api_keys),
            'current_key_index': self.current_key_index + 1,
            'current_key_prefix': self.current_api_key[:20] + '...',
            'failure_counts': {
                f'key_{i+1}': count 
                for i, count in self.key_failure_count.items()
            },
            'healthy_keys': sum(1 for count in self.key_failure_count.values() if count < 5)
        }
    
    def reset_key_failures(self):
        """Reset failure counts for all API keys"""
        self.key_failure_count = {i: 0 for i in range(len(self.api_keys))}
        logger.info("Reset all API key failure counts")
    
    def batch_classify(
        self,
        articles: List[Dict[str, Any]],
        batch_size: int = 1,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple articles (currently sequential)
        
        Args:
            articles: List of article dicts with 'title', 'abstract', 'relevant_paths'
            batch_size: Batch size (currently only supports 1)
            **generation_kwargs: Generation parameters
            
        Returns:
            List of classification results
        """
        results = []
        
        for article in articles:
            result = self.classify_article(
                title=article['title'],
                abstract=article['abstract'],
                relevant_paths=article['relevant_paths'],
                **generation_kwargs
            )
            results.append(result)
        
        return results


def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize classifier (API key from environment)
    classifier = LLMClassifier(
        model_name="gemini-2.5-flash-lite"  # Options: gemini-2.0-flash-exp, gemini-2.5-pro, gemini-1.5-flash, gemini-1.5-pro
    )
    
    # Example article
    title = "Deep Learning for Medical Image Segmentation"
    abstract = """
    This paper presents a novel deep learning approach for automated medical 
    image segmentation. We develop a convolutional neural network architecture 
    that achieves state-of-the-art performance on multiple medical imaging datasets. 
    The model is evaluated on CT and MRI scans for tumor detection and achieves 
    significant improvements over existing methods.
    """
    
    # Example retrieved paths
    relevant_paths = [
        "Natural Science > Computer and Information Science > Artificial Intelligence > Machine Learning > Deep Learning",
        "Medical and Health Science > Basic Medicine > Pathology > Diagnostic Pathology",
        "Natural Science > Computer and Information Science > Artificial Intelligence > Computer Vision > Image Segmentation",
        "Engineering and Technology > Medical Engineering > Biomedical Engineering > Medical Imaging",
        "Medical and Health Science > Clinical Medicine > Radiology > Medical Imaging",
    ]
    
    # Classify
    print("\n=== Classifying Article ===")
    result = classifier.classify_article(
        title=title,
        abstract=abstract,
        relevant_paths=relevant_paths,
        temperature=0.3
    )
    
    # Display result
    print(f"\nClassified Path: {result['classification']['path']}")
    print(f"Confidence: {result['classification']['confidence']}")
    print(f"Reasoning: {result['classification']['reasoning']}")
    print(f"\nPrompt length: {result['prompt_length']} chars")
    print(f"Response length: {result['response_length']} chars")


if __name__ == "__main__":
    main()
