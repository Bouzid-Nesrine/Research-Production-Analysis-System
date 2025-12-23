"""
LLM Classifier - Interface for Qwen 2.5 Instruct 14B
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List
import torch
import logging
import re

logger = logging.getLogger(__name__)


class LLMClassifier:
    """Qwen 2.5 Instruct 14B classifier for taxonomy classification"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        device_map: str = "auto",
        load_in_8bit: bool = False,
        torch_dtype: str = "auto"
    ):
        """
        Initialize LLM classifier
        
        Args:
            model_name: Hugging Face model name
            device_map: Device mapping strategy
            load_in_8bit: Use 8-bit quantization
            torch_dtype: Torch data type
        """
        self.model_name = model_name
        
        logger.info(f"Loading LLM: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        load_kwargs = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        
        if torch_dtype == "auto":
            load_kwargs["torch_dtype"] = "auto"
        else:
            load_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        logger.info("LLM loaded successfully")
    
    def create_classification_prompt(
        self,
        title: str,
        abstract: str,
        relevant_paths: List[str],
        include_reasoning: bool = True
    ) -> str:
        """
        Create optimized prompt for classification
        
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
        
        # Build prompt
        prompt = f"""You are an expert research article classifier. Your task is to classify the given article into the most appropriate category from the provided taxonomy paths.

Article Title: {title}

Article Abstract: {abstract}

Relevant Taxonomy Paths (ranked by relevance):
{paths_text}

Instructions:
1. Carefully analyze the article's title and abstract to understand its main research focus
2. Compare the article's content with each taxonomy path
3. Select EXACTLY ONE path that best represents the article's primary research area
4. The path must be chosen from the list above
5. Provide your selection in the exact format specified below

Response Format:
Path: [Copy the complete path exactly as shown above]
Confidence: [High/Medium/Low]"""
        
        if include_reasoning:
            prompt += "\nReasoning: [Brief explanation in 1-2 sentences why this path was chosen]\n"
        
        prompt += "\nYour classification:"
        
        return prompt
    
    def classify(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate classification using LLM
        
        Args:
            prompt: Classification prompt
            temperature: Sampling temperature (lower = more deterministic)
            max_new_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            LLM response text
        """
        # Format as chat
        messages = [
            {
                "role": "system",
                "content": "You are an expert research article classifier with deep knowledge across all scientific domains."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return response.strip()
    
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
    logger.basicConfig(level=logging.INFO)
    
    # Initialize classifier
    classifier = LLMClassifier(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        load_in_8bit=False  # Set to True if GPU memory limited
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
