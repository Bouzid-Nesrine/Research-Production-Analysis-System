"""
Example: How to use the fine-tuned model in your own code
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str, base_model_name: str):
    """
    Load a fine-tuned model.
    
    Args:
        model_path: Path to fine-tuned model (e.g., "./output/phi-2-classification/final_model")
        base_model_name: Base model name (e.g., "microsoft/phi-2")
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer


def classify_article(model, tokenizer, title: str, abstract: str) -> str:
    """
    Classify a research article.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        title: Article title
        abstract: Article abstract
    
    Returns:
        Classification path
    """
    # Format prompt
    instruction = "Classify the following research article into its appropriate taxonomy path based on the title and abstract. Provide the full classification path from field to subfield."
    input_text = f"Title: {title}\n\nAbstract: {abstract}"
    prompt = f"{instruction}\n\n{input_text}\n\nClassification:"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract classification
    if "Classification:" in generated_text:
        classification = generated_text.split("Classification:")[-1].strip()
    else:
        classification = generated_text.strip()
    
    return classification


def main():
    """Example usage."""
    
    # Configuration
    MODEL_PATH = "./output/phi-2-classification/final_model"
    BASE_MODEL = "microsoft/phi-2"
    
    # Load model (do this once)
    model, tokenizer = load_model(MODEL_PATH, BASE_MODEL)
    
    # Example 1: Single article
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Article")
    print("="*60 + "\n")
    
    title1 = "Graphene-based nanomaterials for high-performance supercapacitors"
    abstract1 = "We synthesized graphene oxide-based composite materials with enhanced electrochemical properties. The materials show excellent specific capacitance and cycling stability, making them promising for energy storage applications."
    
    classification1 = classify_article(model, tokenizer, title1, abstract1)
    print(f"Title: {title1}")
    print(f"Classification: {classification1}\n")
    
    # Example 2: Batch processing
    print("="*60)
    print("EXAMPLE 2: Batch Processing")
    print("="*60 + "\n")
    
    articles = [
        {
            "title": "Machine learning for drug discovery",
            "abstract": "We developed a deep learning model to predict drug-target interactions..."
        },
        {
            "title": "CRISPR-based gene editing in plants",
            "abstract": "We applied CRISPR/Cas9 technology to modify crop genomes for improved traits..."
        },
        {
            "title": "Quantum computing algorithms for optimization",
            "abstract": "We present novel quantum algorithms that outperform classical methods..."
        }
    ]
    
    for i, article in enumerate(articles, 1):
        classification = classify_article(model, tokenizer, article['title'], article['abstract'])
        print(f"{i}. {article['title'][:50]}...")
        print(f"   → {classification}\n")
    
    print("="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
