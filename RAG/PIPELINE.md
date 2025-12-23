# RAG-Based Taxonomy Classification Pipeline

## Overview
This pipeline implements a Retrieval-Augmented Generation (RAG) system to efficiently classify research articles into a hierarchical taxonomy using ChromaDB for vector storage and Qwen 2.5 Instruct 14B for classification.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG Classification Pipeline                  │
└─────────────────────────────────────────────────────────────────┘

1. OFFLINE PHASE (One-time Setup)
   ┌──────────────────────────────────────────────────────────────┐
   │ Taxonomy Preparation                                          │
   │ • Extract all paths from taxonomy (700+ paths)               │
   │ • Generate embeddings for each path                          │
   │ • Store in ChromaDB with metadata                            │
   └──────────────────────────────────────────────────────────────┘

2. ONLINE PHASE (Per Article Classification)
   ┌──────────────────────────────────────────────────────────────┐
   │ Step 1: Article Embedding                                     │
   │ • Combine title + abstract                                   │
   │ • Generate embedding                                         │
   └──────────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────────┐
   │ Step 2: Semantic Retrieval                                    │
   │ • Query ChromaDB with article embedding                      │
   │ • Retrieve top-k (5-10) most similar paths                   │
   │ • Include similarity scores                                  │
   └──────────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────────┐
   │ Step 3: LLM Classification                                    │
   │ • Format prompt with article + relevant paths                │
   │ • Query Qwen 2.5 Instruct 14B                               │
   │ • Extract classification result                              │
   └──────────────────────────────────────────────────────────────┘
                              ↓
   ┌──────────────────────────────────────────────────────────────┐
   │ Step 4: Validation & Output                                   │
   │ • Validate path exists in taxonomy                           │
   │ • Return classification with confidence                      │
   └──────────────────────────────────────────────────────────────┘
```

## Detailed Implementation Steps

### Phase 1: Setup and Preparation

#### 1.1 Environment Setup
```bash
# Install required packages
pip install chromadb sentence-transformers transformers torch accelerate
```

#### 1.2 Extract Taxonomy Paths
- Parse `preprocessed_taxonomy.json`
- Generate full hierarchical paths for each leaf node
- Format: "Domain > Field > Subfield > Specialty > Topic"
- Example: "Natural Science > Computer and Information Science > Artificial Intelligence > Machine Learning > Deep Learning"

#### 1.3 Generate Path Descriptions
For better semantic matching, create rich descriptions:
- **Path**: Full hierarchical path
- **Description**: Contextual description combining all levels
- **Keywords**: Extracted from each level
- **Level**: Depth in taxonomy (1-5)

### Phase 2: Vector Database Setup (ChromaDB)

#### 2.1 Initialize ChromaDB
```python
import chromadb
from chromadb.config import Settings

# Create persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection with metadata
collection = client.get_or_create_collection(
    name="taxonomy_paths",
    metadata={"description": "Research taxonomy classification paths"}
)
```

#### 2.2 Embedding Model Selection
**Recommended Models:**
- `all-MiniLM-L6-v2` (fast, 384 dims) - Good for quick prototyping
- `all-mpnet-base-v2` (768 dims) - Better quality
- `sentence-transformers/allenai-specter` - Scientific papers optimized

```python
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2')
```

#### 2.3 Populate ChromaDB
For each taxonomy path:
1. Generate embedding
2. Store with metadata (path components, level, domain)
3. Create unique ID

```python
# Example structure
for path_id, path_info in taxonomy_paths.items():
    embedding = embedding_model.encode(path_info['description'])
    
    collection.add(
        embeddings=[embedding],
        documents=[path_info['description']],
        metadatas=[{
            'path': path_info['full_path'],
            'domain': path_info['domain'],
            'level': path_info['level'],
            'keywords': ','.join(path_info['keywords'])
        }],
        ids=[path_id]
    )
```

### Phase 3: Article Processing

#### 3.1 Article Preparation
```python
def prepare_article_text(title, abstract):
    """Combine title and abstract for embedding"""
    return f"Title: {title}\n\nAbstract: {abstract}"
```

#### 3.2 Semantic Retrieval
```python
def retrieve_relevant_paths(article_text, top_k=10):
    """Retrieve most relevant taxonomy paths"""
    
    # Generate article embedding
    article_embedding = embedding_model.encode(article_text)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[article_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    
    return results
```

**Optimization Parameters:**
- `top_k`: Start with 10, tune between 5-15
- Consider diversity: Select from different domains if scores are close
- Filter by similarity threshold (e.g., distance < 0.7)

### Phase 4: LLM Integration (Qwen 2.5 Instruct 14B)

#### 4.1 Model Loading
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-14B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
```

#### 4.2 Prompt Engineering
```python
def create_classification_prompt(article_title, article_abstract, relevant_paths):
    """Create optimized prompt for classification"""
    
    paths_text = "\n".join([f"{i+1}. {path}" for i, path in enumerate(relevant_paths)])
    
    prompt = f"""You are a research article classifier. Your task is to classify the given article into the most appropriate category from the provided taxonomy paths.

Article Title: {article_title}

Article Abstract: {article_abstract}

Relevant Taxonomy Paths (from most to least relevant):
{paths_text}

Instructions:
1. Carefully read the article title and abstract
2. Analyze which taxonomy path best represents the article's primary research focus
3. Select EXACTLY ONE path from the list above
4. Provide the complete hierarchical path

Response format:
Path: [Your selected path]
Confidence: [High/Medium/Low]
Reasoning: [Brief explanation in 1-2 sentences]

Your classification:"""
    
    return prompt
```

#### 4.3 LLM Inference
```python
def classify_with_llm(prompt):
    """Query Qwen model for classification"""
    
    messages = [
        {"role": "system", "content": "You are an expert research article classifier."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        temperature=0.3,  # Lower for more deterministic results
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.batch_decode(
        generated_ids[:, model_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]
    
    return response
```

### Phase 5: Post-Processing and Validation

#### 5.1 Parse LLM Response
```python
def parse_classification_response(response):
    """Extract structured data from LLM response"""
    
    import re
    
    path_match = re.search(r'Path:\s*(.+)', response)
    confidence_match = re.search(r'Confidence:\s*(\w+)', response)
    reasoning_match = re.search(r'Reasoning:\s*(.+)', response)
    
    return {
        'path': path_match.group(1).strip() if path_match else None,
        'confidence': confidence_match.group(1).strip() if confidence_match else None,
        'reasoning': reasoning_match.group(1).strip() if reasoning_match else None
    }
```

#### 5.2 Validation
```python
def validate_classification(classified_path, original_taxonomy):
    """Verify path exists in taxonomy"""
    
    # Check if path exists
    # Return validated result with fallback options
    
    return {
        'valid': True/False,
        'validated_path': classified_path,
        'alternative_paths': []  # If validation fails
    }
```

## Complete Pipeline Function

```python
def classify_article(title, abstract, top_k=10):
    """
    Complete RAG-based classification pipeline
    
    Args:
        title: Article title
        abstract: Article abstract
        top_k: Number of paths to retrieve (5-15)
    
    Returns:
        Classification result with metadata
    """
    
    # Step 1: Prepare article text
    article_text = prepare_article_text(title, abstract)
    
    # Step 2: Retrieve relevant paths
    retrieval_results = retrieve_relevant_paths(article_text, top_k)
    
    relevant_paths = [
        metadata['path'] 
        for metadata in retrieval_results['metadatas'][0]
    ]
    
    similarity_scores = retrieval_results['distances'][0]
    
    # Step 3: Create prompt and classify
    prompt = create_classification_prompt(title, abstract, relevant_paths)
    llm_response = classify_with_llm(prompt)
    
    # Step 4: Parse and validate
    parsed_result = parse_classification_response(llm_response)
    validated_result = validate_classification(parsed_result['path'], taxonomy)
    
    # Step 5: Return complete result
    return {
        'article': {
            'title': title,
            'abstract': abstract
        },
        'classification': {
            'path': validated_result['validated_path'],
            'confidence': parsed_result['confidence'],
            'reasoning': parsed_result['reasoning']
        },
        'retrieval_metadata': {
            'retrieved_paths': relevant_paths,
            'similarity_scores': similarity_scores,
            'top_k': top_k
        },
        'valid': validated_result['valid']
    }
```

## Performance Optimizations

### 1. Token Reduction
- **Before**: ~700 paths × 20 tokens = 14,000+ tokens
- **After**: ~10 paths × 20 tokens = 200 tokens
- **Savings**: ~98.5% token reduction

### 2. Embedding Caching
```python
# Cache article embeddings for batch processing
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_article_embedding(article_text):
    return embedding_model.encode(article_text)
```

### 3. Batch Processing
```python
def batch_classify_articles(articles, batch_size=8):
    """Process multiple articles efficiently"""
    
    results = []
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        
        # Batch embed
        texts = [prepare_article_text(a['title'], a['abstract']) for a in batch]
        embeddings = embedding_model.encode(texts, batch_size=batch_size)
        
        # Process each
        for article, embedding in zip(batch, embeddings):
            result = classify_article(article['title'], article['abstract'])
            results.append(result)
    
    return results
```

### 4. GPU Optimization
```python
# Load models with optimal settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    load_in_8bit=True,  # Use quantization if needed
)
```

## Evaluation Metrics

### 1. Retrieval Quality
- **Recall@K**: Does correct path appear in top-K?
- **MRR**: Mean Reciprocal Rank of correct path
- **nDCG**: Normalized Discounted Cumulative Gain

### 2. Classification Accuracy
- **Exact Match**: Full path matches ground truth
- **Hierarchical Accuracy**: Matches at each level
- **Domain Accuracy**: Top-level domain correct

### 3. Efficiency Metrics
- **Inference Time**: Total time per article
- **Token Usage**: Average tokens per classification
- **Throughput**: Articles per second

## Hyperparameter Tuning

### Key Parameters to Tune:
1. **top_k** (5-15): Balance between context and noise
2. **temperature** (0.1-0.7): Lower = more deterministic
3. **similarity_threshold**: Filter low-relevance paths
4. **embedding_model**: Trade-off between speed and quality

### Recommended Starting Points:
```python
CONFIG = {
    'top_k': 10,
    'temperature': 0.3,
    'max_new_tokens': 256,
    'similarity_threshold': 0.7,
    'embedding_model': 'all-mpnet-base-v2'
}
```

## Error Handling

```python
def safe_classify(title, abstract, max_retries=3):
    """Classification with error handling"""
    
    for attempt in range(max_retries):
        try:
            result = classify_article(title, abstract)
            
            if not result['valid']:
                # Fallback to more paths
                result = classify_article(title, abstract, top_k=20)
            
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    'error': str(e),
                    'fallback': 'manual_review_required'
                }
            continue
```

## Monitoring and Logging

```python
import logging
from datetime import datetime

logging.basicConfig(
    filename=f'rag_classification_{datetime.now():%Y%m%d}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_classification(article_id, result):
    """Log classification for analysis"""
    
    logging.info(f"""
    Article ID: {article_id}
    Path: {result['classification']['path']}
    Confidence: {result['classification']['confidence']}
    Top Retrieved: {result['retrieval_metadata']['retrieved_paths'][0]}
    Valid: {result['valid']}
    """)
```

## Expected Performance Improvements

| Metric | Before (Full Taxonomy) | After (RAG) | Improvement |
|--------|----------------------|-------------|-------------|
| Avg Tokens | 14,000+ | 200-500 | 96-98% ↓ |
| Inference Time | 30-60s | 5-10s | 80-85% ↓ |
| Accuracy | Baseline | +5-15% | Better |
| Cost per 1K | $X | $0.02X | 98% ↓ |

## Deployment Considerations

### 1. Production Setup
- Use persistent ChromaDB storage
- Implement API endpoints
- Add request queuing
- Enable result caching

### 2. Scaling
- Multiple GPU workers for LLM
- Separate embedding service
- Load balancing

### 3. Maintenance
- Regular taxonomy updates
- Re-embedding on changes
- Performance monitoring
- A/B testing for improvements

## Next Steps

1. ✅ Set up environment
2. ✅ Extract taxonomy paths
3. ✅ Initialize ChromaDB
4. ✅ Generate embeddings
5. ✅ Load Qwen model
6. ✅ Test on sample articles
7. ✅ Tune hyperparameters
8. ✅ Evaluate performance
9. ✅ Deploy to production
