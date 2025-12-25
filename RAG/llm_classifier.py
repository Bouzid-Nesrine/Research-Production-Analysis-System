"""
LLM Classifier - Interface for Google Gemini via AI Studio API
"""

import os
import google.generativeai as genai
from typing import Dict, Any, Optional, List
import logging
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LLMClassifier:
    """Google Gemini classifier using AI Studio API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        api_base_url: Optional[str] = None
    ):
        """
        Initialize LLM classifier with Google AI Studio API
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env variable)
            model_name: Model name (gemini-2.0-flash-exp, gemini-2.5-pro, gemini-1.5-flash, gemini-1.5-pro)
            api_base_url: Not used (kept for compatibility)
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model_name = model_name
        
        # Configure Google Generative AI
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"Initialized LLM classifier with model: {model_name}")
    
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
        **kwargs
    ) -> str:
        """
        Generate classification using Google AI Studio API
        
        Args:
            prompt: Classification prompt
            temperature: Sampling temperature (lower = more deterministic, 0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0-1)
            **kwargs: Additional API parameters
            
        Returns:
            LLM response text
        """
        # Build system message + user prompt
        full_prompt = "You are an expert research article classifier with deep knowledge across all scientific domains.\n\n" + prompt
        
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
            
            # Extract text
            return response.text.strip()
                
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
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
