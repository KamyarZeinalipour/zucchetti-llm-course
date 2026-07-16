# 🔬 Lab 01: Embeddings Exploration

> **Lecture:** 1 — LLM & Embeddings Fundamentals
> **Duration:** 35 minutes
> **Difficulty:** ⭐ (Beginner)

## 🎯 Objectives

By the end of this lab you will:
- Generate embeddings for categorized terms using a local model
- Compute pairwise cosine similarity to measure semantic closeness
- Visualize how related terms cluster together in vector space
- Compare a multilingual model vs. an English-only model

## 📋 Prerequisites

| Requirement | How to Check |
|---|---|
| Python 3.10+ | `python --version` |
| Dependencies installed | `pip install sentence-transformers numpy scikit-learn rich` |

> **No API key needed** — this lab runs 100% locally using sentence-transformers.

## 🚀 Quick Start

```bash
cd lab_01_embeddings
python lab_01_embeddings.py
```

## 📂 Files

| File | Description |
|---|---|
| `lab_01_embeddings.py` | Main lab script with TODO sections for you to complete |
| `lab_01_embeddings.ipynb` | Jupyter notebook version with explanations |
| `solutions.py` | Complete solutions (don't peek until you've tried!) |
| `data/terms_it.json` | 20 categorized terms with Italian/English translations |

## 🔧 Tasks

### Task 1: Load and Explore the Dataset
The script loads 20 terms across 5 categories (with Italian and English translations).

> No code changes needed — just observe the output.

### Task 2: Generate Embeddings
The script generates embeddings using `paraphrase-multilingual-MiniLM-L12-v2`. Each term becomes a 384-dimensional vector.

> No code changes needed — observe the embedding shape.

### Task 3: Implement Cosine Similarity ✍️
**Your turn!** Find the `compute_similarity_matrix()` function and implement it.

**Hint:** You can use scikit-learn:
```python
from sklearn.metrics.pairwise import cosine_similarity
return cosine_similarity(embeddings)
```

Or implement it manually with numpy:
```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized = embeddings / norms
return np.dot(normalized, normalized.T)
```

### Task 4: Find Top Similar Pairs ✍️
**Your turn!** Find the `find_top_similar_pairs()` function and implement the pair-finding logic.

**Hint:** Loop over the upper triangle of the similarity matrix (where `i < j`), collect all pairs with their scores, sort by score, and return the top K.

### Task 5: Interpret Results
Look at the output:
- Do terms in the same category have higher similarity?
- Which cross-category pairs are surprisingly similar? Why?

### Task 6: Bonus — Compare Models
Uncomment the `compare_models()` call in `main()` to compare:
- `paraphrase-multilingual-MiniLM-L12-v2` (multilingual)
- `all-MiniLM-L6-v2` (English-only)

> Which model produces better clusters?

## ✅ Expected Output

When implemented correctly, you should see:
1. A colored similarity matrix in the terminal
2. A table of the top 10 most similar pairs
3. Most top pairs should be from the same category (e.g., two drugs, two diseases)

Example output:
```
Top 10 Most Similar Term Pairs
┌───┬────────────────────┬────────────────────┬────────────┬────────────────┐
│ # │ Term 1 (IT)        │ Term 2 (IT)        │ Similarity │ Same Category? │
├───┼────────────────────┼────────────────────┼────────────┼────────────────┤
│ 1 │ paracetamolo       │ ibuprofene         │ 0.82       │ ✓ Yes          │
│ 2 │ risonanza magnetica│ tomografia comput. │ 0.79       │ ✓ Yes          │
│ 3 │ diabete mellito    │ glicemia a digiuno │ 0.71       │ ✗ No           │
│...│ ...                │ ...                │ ...        │ ...            │
└───┴────────────────────┴────────────────────┴────────────┴────────────────┘

Observation: 8/10 of the most similar pairs are in the same category
```

## 💡 Going Further

- Try adding your own terms to the JSON file
- Experiment with different embedding models from [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- Try encoding the English translations instead of Italian — how do the results change?
- Visualize the embeddings in 2D using t-SNE or UMAP

## 🆘 Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: sentence_transformers` | `pip install sentence-transformers` |
| Model download slow | First run downloads ~120MB model. Be patient. |
| Memory error | Close other applications. The model needs ~500MB RAM. |
| All zeros in similarity matrix | You need to implement the TODO in Step 3! |
