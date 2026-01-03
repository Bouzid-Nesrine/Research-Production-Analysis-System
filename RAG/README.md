# RAG-Based Taxonomy Classification System

Efficient research article classification using Retrieval-Augmented Generation (RAG) with ChromaDB and a fine-tuned SciBERT model.

##  Overview

This system classifies research articles into a hierarchical taxonomy using:
- **Vector Database (ChromaDB)**: Stores 1,449 taxonomy paths as embeddings
- **Semantic Retrieval**: Finds top-k most relevant paths for each article
- **Fine-tuned SciBERT Model**: Re-ranks candidates using domain-specific knowledge
- **LoRA Adaptation**: Efficient fine-tuning on 862 scientific domain classes
- **Local Inference**: CPU-based inference, no API costs or rate limits

## 📊 Performance Benefits

| Metric | Without RAG | With RAG | Improvement |
|--------|-------------|----------|-------------|
| Accuracy (F1) | 27.00% | 34.54% | **+7.54%** |
| Relative Improvement | - | - | **+27.9%** |
| Inference time | ~0.15s | ~0.85s | Per article |
| Retrieval Recall@5 | N/A | 41.00% | - |
| Re-ranking Effectiveness | N/A | 61.8% | When correct in top-5 |
| Cost | Free | Free | **Local inference** |

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to RAG directory
cd RAG

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Fine-tuned Model

The system uses a fine-tuned SciBERT model with LoRA adapters:

```bash
# Model is located in:
# RAG/best_models/scibert_lora_final/
# (Should be already present in the repository)
```

### 3. Setup Database

```bash
# Initialize ChromaDB with taxonomy paths (one-time setup)
python setup_pipeline.py

# To reset and rebuild database
python setup_pipeline.py --reset
```

This will:
- Parse taxonomy into 4500+ hierarchical paths
- Generate embeddings using sentence-transformers (CPU-friendly)
- Store in ChromaDB for fast retrieval

### 4. Classify Articles

#### Python Script

```python
from rag_pipeline import RAGClassificationPipeline

# Initialize pipeline (uses local fine-tuned SciBERT model)
pipeline = RAGClassificationPipeline(auto_setup=True)

# Classify single article
result = pipeline.classify_article(
    title="Deep Learning for Image Classification",
    abstract="This paper presents a novel CNN architecture..."
)

print(f"Path: {result['classification']['path']}")
print(f"Confidence: {result['classification']['confidence']}")
print(f"Model Score: {result['classification']['model_score']}")
```

#### Batch Processing

```python
articles = [
    {'title': '...', 'abstract': '...'},
    {'title': '...', 'abstract': '...'},
]

results = pipeline.batch_classify(articles, show_progress=True)
pipeline.save_results(results, 'output/results.json')
```

#### Jupyter Notebook

```bash
jupyter notebook RAG_Classification_Demo.ipynb
```

## 📁 Project Structure

```
RAG/
├── config.py                   # Configuration settings
├── taxonomy_parser.py          # Parse taxonomy into paths
├── vector_db_manager.py        # ChromaDB interface
├── local_model_classifier.py   # Fine-tuned SciBERT classifier
├── rag_pipeline.py             # Complete pipeline
├── setup_pipeline.py           # Database setup script
├── evaluate_rag_accuracy.py    # Evaluation script
├── requirements.txt            # Dependencies
├── EVALUATION_REPORT.md        # Performance analysis
├── PIPELINE.md                 # Detailed documentation
├── README.md                   # This file
│
├── best_models/
│   └── scibert_lora_final/     # Fine-tuned SciBERT with LoRA
├── chroma_db/                  # ChromaDB storage (created after setup)
├── logs/                       # Classification logs
├── evaluation_results/         # Evaluation outputs
└── taxonomy_paths.json         # Extracted paths (created after setup)
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
RAG_CONFIG = {
    "top_k": 5,                      # Paths to retrieve (5-15)
    "model_weight": 0.6,             # Weight for model score (0.6)
    "retrieval_weight": 0.4,         # Weight for retrieval similarity (0.4)
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Embedding model (384 dims)
LOCAL_MODEL_PATH = "best_models/scibert_lora_final"  # Fine-tuned SciBERT with LoRA
```

## 📖 Detailed Workflow

### Phase 1: Offline Setup (One-time)

1. **Parse Taxonomy** (`taxonomy_parser.py`)
   - Extract all hierarchical paths from JSON
   - Generate rich descriptions for each path
   - Extract keywords for better matching

2. **Generate Embeddings** (`vector_db_manager.py`)
   - Use sentence-transformers to embed path descriptions
   - Store in ChromaDB with metadata

### Phase 2: Online Classification (Per Article)

1. **Retrieve Relevant Paths** (~0.024s)
   - Embed article (title + abstract)
   - Query ChromaDB for top-k similar paths (k=5)
   - Return similarity scores

2. **Neural Re-ranking** (~0.83s)
   - For each candidate path, create input: `[Article] [SEP] [Path]`
   - Score with fine-tuned SciBERT model
   - Combine: 0.6 × model_score + 0.4 × retrieval_similarity

3. **Final Selection**
   - Select path with highest combined score
   - Return classification with confidence and scores

## 🎛️ Advanced Usage

### Custom Embedding Models

```python
# For scientific papers (better accuracy)
pipeline = RAGClassificationPipeline(
    embedding_model="sentence-transformers/allenai-specter"
)

# For faster processing
pipeline = RAGClassificationPipeline(
    embedding_model="all-MiniLM-L6-v2"
)
```

### Score Fusion Tuning

```python
# Adjust model vs retrieval weights
from local_model_classifier import LocalModelClassifier

classifier = LocalModelClassifier()
result = classifier.classify_with_paths(
    title=title,
    abstract=abstract,
    candidate_paths=retrieved_paths,
    model_weight=0.7,      # Emphasize model more
    retrieval_weight=0.3   # De-emphasize retrieval
)
```

### Hyperparameter Tuning

```python
# Test different top_k values
for k in [5, 10, 15, 20]:
    result = pipeline.classify_article(
        title=title,
        abstract=abstract,
        top_k=k
    )
    # Evaluate results...

# Test different score weights
for model_w in [0.5, 0.6, 0.7, 0.8]:
    retrieval_w = 1.0 - model_w
    result = classifier.classify_with_paths(
        title=title,
        abstract=abstract,
        candidate_paths=paths,
        model_weight=model_w,
        retrieval_weight=retrieval_w
    )
```

### Filtering by Domain

```python
# Retrieve only from specific domain
results = db_manager.retrieve_relevant_paths(
    query_text=article_text,
    top_k=10,
    filter_domain="Natural Science"
)
```

## 📊 Evaluation

### Run on Test Set

```python
# Load your test data
test_articles = [...]  # with ground truth labels

results = []
for article in test_articles:
    result = pipeline.classify_article(
        title=article['title'],
        abstract=article['abstract']
    )
    results.append({
        'predicted': result['classification']['path'],
        'ground_truth': article['true_path'],
        'correct': result['classification']['path'] == article['true_path']
    })

# Calculate metrics
accuracy = sum(r['correct'] for r in results) / len(results)
print(f"Accuracy: {accuracy:.2%}")
```

### Evaluation Metrics

```python
from sklearn.metrics import classification_report

y_true = [r['ground_truth'] for r in results]
y_pred = [r['predicted'] for r in results]

print(classification_report(y_true, y_pred))
```

## 🔍 Monitoring and Logging

Logs are saved to `logs/rag_classification_YYYYMMDD.log`:

```python
# View statistics
stats = pipeline.get_statistics()
print(f"Total classified: {stats['total_classified']}")
print(f"Success rate: {stats['success_rate']:.2%}")
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Option 1: Enable 8-bit quantization
LLM_LOAD_CONFIG = {"load_in_8bit": True}

# Option 2: Use smaller embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Option 3: Process in smaller batches
results = pipeline.batch_classify(articles, batch_size=1)
```

### ChromaDB Not Found

```bash
# Rebuild database
python setup_pipeline.py --reset
```

### Slow Retrieval

```python
# Reduce top_k
RAG_CONFIG['top_k'] = 5

# Or use faster embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
```

### Invalid Classifications

```python
# Increase top_k to provide more options
result = pipeline.classify_article(
    title=title,
    abstract=abstract,
    top_k=20
)

# Lower temperature for more deterministic results
result = pipeline.classify_article(
    title=title,
    abstract=abstract,
    temperature=0.1
)
```

## 📚 Documentation

- **[PIPELINE.md](PIPELINE.md)**: Comprehensive technical documentation
- **[config.py](config.py)**: All configuration options
- **Code docstrings**: Every function is documented

## 🧪 Example Results

```json
{
  "article": {
    "title": "Deep Learning for Medical Image Segmentation",
    "abstract": "This paper presents a novel CNN architecture..."
  },
  "classification": {
    "path": "Natural Science > Computer and Information Science > Artificial Intelligence > Computer Vision > Image Segmentation",
    "confidence": "High",
    "reasoning": "The article focuses on applying deep learning to medical image segmentation, which is a core computer vision task.",
    "valid": true
  },
  "metadata": {
    "retrieval": {
      "top_k": 10,
      "retrieval_time": 0.15,
      "similarity_scores": [0.85, 0.82, 0.79, ...]
    },
    "classification": {
      "classification_time": 4.5,
      "prompt_length": 450,
      "response_length": 180
    },
    "total_time": 4.65
  }
}
```

## 🚦 Performance Tips

1. **Batch Processing**: Process articles in batches for better throughput
2. **Cache Results**: Enable result caching for repeated queries
3. **Optimal top_k**: Start with 10, tune based on your data (5-15 range)
4. **GPU Utilization**: Use CUDA for both embedding and LLM inference
5. **Embedding Cache**: Embeddings are cached automatically (LRU cache)

## 📦 Requirements

- Python 3.8+
- CPU-only inference supported (no GPU required)
- 8GB+ RAM
- Dependencies: transformers, peft, torch, sentence-transformers, chromadb

