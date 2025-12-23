# Step-by-Step RAG Testing Guide

This folder contains step-by-step test scripts to verify each component of the RAG system independently.

## 🧪 Test Scripts

### Step 1: Taxonomy Parser
**File:** `test_1_taxonomy_parser.py`

**What it tests:**
- Loading taxonomy JSON file
- Extracting hierarchical paths
- Path structure and metadata
- Domain and level distribution

**Run:**
```bash
python test_1_taxonomy_parser.py
```

**Expected output:**
- ✓ Taxonomy loaded successfully
- ✓ ~700 paths extracted
- ✓ Statistics about domains and levels
- ✓ Sample paths displayed
- ✓ Saves `test_taxonomy_paths.json`

---

### Step 2: Vector Database (ChromaDB)
**File:** `test_2_vector_database.py`

**What it tests:**
- ChromaDB initialization
- Database population with embeddings
- Collection statistics
- Basic retrieval functionality
- Performance metrics

**Run:**
```bash
python test_2_vector_database.py
```

**Expected output:**
- ✓ ChromaDB initialized at `chroma_db/`
- ✓ All paths embedded and stored
- ✓ Semantic retrieval working
- ✓ Query performance measured
- ⏱️ Takes 2-5 minutes on first run (generates embeddings)

---

### Step 3: Embeddings Quality
**File:** `test_3_embeddings.py`

**What it tests:**
- Embedding model loading
- Embedding generation
- Semantic similarity calculation
- Consistency checks
- Batch encoding performance

**Run:**
```bash
python test_3_embeddings.py
```

**Expected output:**
- ✓ Embeddings generated correctly
- ✓ Semantic relationships captured
- ✓ Similarity scores look reasonable
- ✓ Batch encoding is faster than single
- ✓ Saves visualization (if matplotlib available)

---

### Step 4: Retrieval Quality
**File:** `test_4_retrieval.py`

**What it tests:**
- Semantic search quality
- Multiple query types
- Real article retrieval
- Top-k parameter effects
- Similarity thresholds
- Domain diversity

**Run:**
```bash
python test_4_retrieval.py
```

**Expected output:**
- ✓ Retrieval works for various queries
- ✓ Similarity scores analyzed
- ✓ Results exported to CSV
- ✓ Visualization saved (if matplotlib available)

---

## 🚀 Quick Start

Run all tests in sequence:

```bash
# Step 1: Taxonomy
python test_1_taxonomy_parser.py

# Step 2: Vector Database (takes 2-5 min first time)
python test_2_vector_database.py

# Step 3: Embeddings
python test_3_embeddings.py

# Step 4: Retrieval
python test_4_retrieval.py
```

## 📊 Generated Files

After running tests, you'll have:

```
RAG/
├── test_taxonomy_paths.json          # Extracted taxonomy paths
├── test_retrieval_results.csv        # Retrieval results
├── test_embeddings_visualization.png # Embedding 2D projection
├── test_retrieval_visualization.png  # Similarity distributions
└── chroma_db/                        # Vector database (persistent)
```

## ✅ Success Indicators

Each test should end with:
```
✅ ALL TESTS PASSED - [Component] is working correctly!
```

If you see this, the component is ready to use!

## 🐛 Troubleshooting

### Test 1 fails: "Taxonomy file not found"
- Check `config.py` → `TAXONOMY_PATH`
- Ensure `preprocessed_taxonomy.json` exists

### Test 2 fails: "Module 'chromadb' not found"
```bash
pip install chromadb sentence-transformers
```

### Test 3 fails: "sklearn not available"
- Visualization skipped, but embeddings still work
- Install: `pip install scikit-learn matplotlib`

### Test 4 slow or fails
- Ensure Test 2 completed successfully
- Database should have ~700 paths

## 📝 What Each Test Validates

| Test | Component | Time | Purpose |
|------|-----------|------|---------|
| 1 | Taxonomy Parser | ~5s | Verify taxonomy structure |
| 2 | Vector Database | 2-5m | Setup ChromaDB with embeddings |
| 3 | Embeddings | ~30s | Verify semantic quality |
| 4 | Retrieval | ~1m | Test search accuracy |

## 🎯 Next Steps

After all tests pass:

1. **Try full pipeline:**
   ```bash
   python quickstart.py
   ```

2. **Interactive demo:**
   ```bash
   jupyter notebook RAG_Classification_Demo.ipynb
   ```

3. **Use in your code:**
   ```python
   from rag_pipeline import RAGClassificationPipeline
   pipeline = RAGClassificationPipeline(auto_setup=True)
   result = pipeline.classify_article(title, abstract)
   ```

## 💡 Tips

- **Run tests in order** - each builds on the previous
- **First run is slow** - embedding generation takes time
- **Subsequent runs are fast** - database is cached
- **Review outputs** - check CSV and visualizations
- **Adjust config.py** - if you need different settings

## 📞 Need Help?

If tests fail:
1. Check error messages carefully
2. Review `config.py` settings
3. Ensure all dependencies installed: `pip install -r requirements.txt`
4. Check that GPU is available (for LLM later)

---

**Ready?** Start with: `python test_1_taxonomy_parser.py`
