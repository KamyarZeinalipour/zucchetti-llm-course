# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🔬 Lab 01: Embeddings Exploration
# **Lecture 1 — LLM & Embeddings Fundamentals**
#
# ---
#
# ## Objective
# Generate embeddings for categorized terms, compute pairwise cosine similarity,
# and visualize how semantically related terms cluster together in vector space.
#
# ## What You'll Learn
# | Concept | Slide Reference |
# |---|---|
# | Embeddings as "GPS coordinates for meaning" | Section 3: What is an Embedding? |
# | Word2Vec: king − man + woman ≈ queen | Section 3: Word2Vec — Where Embeddings Started |
# | Cosine similarity & distance metrics | Section 3: Cosine Similarity + Distance Metrics |
# | BPE tokenization algorithm | Section 2: BPE Algorithm — Step by Step |
# | Why multilingual models matter | Section 4: Cross-Lingual Alignment |
# | How embedding dimensions affect quality | Section 3: Why Dimensions Matter |
# | Transformer architectures (encoder/decoder) | Section 1: Encoder vs Decoder |
#
# ### Theory Slides You Should Know (covered in lecture)
# | Topic | Key Formula |
# |---|---|
# | Multi-Head Attention | MultiHead(Q,K,V) = Concat(heads)·W^O |
# | Positional Encoding | PE(pos,2i) = sin(pos / 10000^(2i/d)) |
# | Cross-Entropy Loss | L(θ) = −Σ log P(xₜ | x₁...xₜ₋₁) |
# | Perplexity | PPL = exp(−1/N · Σ log P) |
# | RLHF | L_RLHF = −E[r(x,y)] + β·KL[π_θ || π_ref] |
#
# ⏱ **Duration:** ~25 minutes
#
# ### How This Notebook Works
# This notebook covers all the hands-on exercises for Lecture 1.
# Each step builds on the previous one — just run the cells in order.
# Steps 1–9 are the core lab. Step 10 is a bonus.

# %% [markdown]
# ---
# ## ☁️ Colab Setup
# If you're running this on **Google Colab**, run the cell below to
# clone the course repository and install dependencies.
# If you're running locally, skip to "Setup & Imports".

# %%
# --- Run this cell in Google Colab ---
!pip install -q sentence-transformers transformers numpy scikit-learn matplotlib pandas tabulate

# Clone the repo and navigate to the lab directory
import os
if not os.path.exists("data/terms_it.json"):
    !git clone -q https://github.com/KamyarZeinalipour/zucchetti-llm-course.git _repo
    os.chdir("_repo/lab_01_embeddings")
    print("✓ Moved to lab_01_embeddings directory")
else:
    print("✓ Data file found — running locally")

# %% [markdown]
# ---
# ## Setup & Imports

# %%
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# %% [markdown]
# ---
# ## Step 1: Load the Terms Dataset
#
# We load **20 terms across 4 categories** from a JSON file.
# Each term has both an English and Italian translation — we'll use the
# **English terms** for the main exercise, and compare English vs Italian
# in the bonus step to see how multilingual models handle cross-lingual input.
#
# > 💡 **From the slides:** Remember that tokenization affects non-English languages
# > differently. Italian text uses more tokens than English for the same meaning.
# > We'll see this effect hands-on below and again in the bonus Step 10.

# %%
def load_terms(data_path: str = None) -> dict:
    """Load the terms dataset from JSON."""
    if data_path is None:
        candidates = [
            Path("data/terms_it.json"),
            Path("lab_01_embeddings/data/terms_it.json"),
        ]
        data_path = next((p for p in candidates if p.exists()), candidates[0])

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

data = load_terms()

# Display as a pandas DataFrame
df_terms = pd.DataFrame([
    {"#": i, "English": t["en"], "Italian": t["it"], "Category": t["category"]}
    for i, t in enumerate(data["terms"], 1)
])
print(f"📋 Terms Dataset — {len(data['terms'])} terms across {len(df_terms['Category'].unique())} categories\n")
df_terms

# %% [markdown]
# ---
# ## Step 2: Quick Tokenization Demo
#
# Before we generate embeddings, let's see **tokenization** in action.
# From the slides: LLMs don't see words — they see **tokens**.
#
# > 📄 **From the slides — BPE Algorithm:**
# > The tokenizer uses **Byte-Pair Encoding** (Sennrich et al., ACL 2016).
# > It starts from characters and merges the most frequent pairs:
# > `[l, o, w, e, r]` → `[lo, w, e, r]` → `[low, er]` → `[lower]`
#
# Key insight: **Non-English text uses more tokens** for the same meaning,
# which means higher cost and less context window.
#
# > 🔗 Try it yourself: [platform.openai.com/tokenizer](https://platform.openai.com/tokenizer)

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def compare_tokens(en: str, it: str):
    """Compare tokenization between English and Italian text."""
    en_tokens = tokenizer.tokenize(en)
    it_tokens = tokenizer.tokenize(it)
    return {
        "English": en,
        "EN tokens": len(en_tokens),
        "EN breakdown": " | ".join(en_tokens),
        "Italian": it,
        "IT tokens": len(it_tokens),
        "IT breakdown": " | ".join(t.replace("##", "·") for t in it_tokens),
        "Extra tokens": f"+{len(it_tokens) - len(en_tokens)}",
    }

examples = [
    ("machine learning",           "apprendimento automatico"),
    ("artificial intelligence",    "intelligenza artificiale"),
    ("investment portfolio",       "portafoglio di investimenti"),
]

df_tokens = pd.DataFrame([compare_tokens(en, it) for en, it in examples])
print("🔍 How the Tokenizer Splits Text")
print("   · = subword boundary (the tokenizer split a word into pieces)")
print("   More tokens = higher API cost + less fits in the context window\n")
df_tokens

# %% [markdown]
# ### 🧪 Try it yourself!
# Add your own English / Italian pair below and re-run:

# %%
# ✏️ Change these and re-run!
pd.DataFrame([compare_tokens("deep learning", "apprendimento profondo")])

# %% [markdown]
# ---
# ## Step 3: Generate Embeddings
#
# An **embedding** converts text into a dense vector (a list of numbers) that
# captures its meaning. Think of it like **GPS coordinates for meaning**:
#
# | GPS | Embeddings |
# |---|---|
# | Rome → [41.9, 12.5] | "machine learning" → [0.23, -0.45, ...] |
# | Naples → [40.8, 14.3] | "deep learning" → [0.21, -0.42, ...] |
# | Close coordinates = close cities | Close vectors = similar meaning |
#
# Instead of 2 dimensions (latitude, longitude), embeddings use **hundreds or thousands
# of dimensions** (depending on the model) to capture nuances of meaning.
#
# ### Encoder vs Decoder — Why It Matters Here
# From the slides, remember there are **3 types** of transformer architectures:
#
# | Architecture | What it does | Examples |
# |---|---|---|
# | **Encoder-only** | Understands text → produces embeddings | BERT, MiniLM ← *we use this* |
# | **Decoder-only** | Generates text token by token | GPT-4, LLaMA, Mistral |
# | **Encoder-Decoder** | Transforms input → output | T5, BART |
#
# Embedding models are **encoder-only** — they read the full text at once and
# compress its meaning into a single vector. That's exactly what we need here.
#
# Let's see all **3 architectures in action** with small examples:

# %%
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import torch

# Load all 3 models once
print("⟳ Loading decoder-only model (GPT-2)...")
generator = pipeline("text-generation", model="gpt2", max_new_tokens=30, do_sample=True, temperature=0.7)

print("⟳ Loading encoder-decoder model (T5)...")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

print("⟳ Loading encoder-only model (MiniLM)...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("✓ All 3 models loaded!\n")


# --- Reusable functions for each architecture ---

def demo_decoder(prompt: str):
    """🟠 Decoder-only (GPT-2): generates text from a prompt."""
    result = generator(prompt)[0]["generated_text"]
    print(f"🟠 Decoder-only (GPT-2) — Generates text token by token")
    print(f"   Input:  \"{prompt}\"")
    print(f"   Output: {result}")
    print(f"   → This is how ChatGPT, LLaMA, and Mistral work\n")


def demo_encoder_decoder(text: str):
    """🟢 Encoder-Decoder (T5): translates text EN → DE."""
    full_input = f"translate English to German: {text}"
    inputs = t5_tokenizer(full_input, return_tensors="pt")
    with torch.no_grad():
        enc_out = t5_model.encoder(**inputs)
    hidden = enc_out.last_hidden_state
    output_ids = t5_model.generate(**inputs, max_length=50)
    translation = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    vals = hidden[0][0][:5].tolist()
    vals_str = ", ".join(f"{v:.4f}" for v in vals)

    print(f"🟢 Encoder-Decoder (T5) — Two-stage: encode then decode")
    print(f"   Input:  \"{text}\"")
    print(f"   ENCODER → {hidden.shape[1]} tokens × {hidden.shape[2]} dims: [{vals_str}, ...]")
    print(f"   DECODER → \"{translation}\"")
    print(f"   → Used for translation, summarization, Q&A\n")


def demo_encoder(text: str):
    """🔵 Encoder-only (MiniLM): produces an embedding vector."""
    emb = embedding_model.encode([text])
    print(f"🔵 Encoder-only (MiniLM) — Understands text → produces embeddings")
    print(f"   Input:  \"{text}\"")
    print(f"   Output: [{emb[0][0]:.4f}, {emb[0][1]:.4f}, {emb[0][2]:.4f}, ...] ({emb.shape[1]} dims)")
    print(f"   → This is what we use for search and similarity! ✅\n")


# --- Run the demo with default inputs ---
print("=" * 60)
print("🔬 Three Transformer Architectures in Action")
print("=" * 60 + "\n")

demo_decoder("Artificial intelligence will")
demo_encoder_decoder("Machine learning is a subset of artificial intelligence.")
demo_encoder("machine learning")

print("💡 The encoder-decoder's hidden states are similar to embeddings!")
print("   The difference: encoder-decoder feeds them to a decoder,")
print("   encoder-only returns them directly for search and similarity.")

# %% [markdown]
# ### 🧪 Try it yourself!
# Change the text below and re-run the cell to see how each architecture responds:

# %%
# ✏️ Change these inputs and re-run!
demo_decoder("The future of healthcare is")

# %%
# ✏️ Try translating different sentences:
demo_encoder_decoder("Deep learning models can understand images and text.")

# %%
# ✏️ Try embedding different terms:
demo_encoder("stock market crash")

# %%
# Free memory from the generation models (we keep embedding_model for later)
del generator, t5_model, t5_tokenizer


# %% [markdown]
# ### Embedding Models Landscape
# From the slides — here are the main options:
#
# | Model | Dimensions | Languages | Best for |
# |---|---|---|---|
# | `all-MiniLM-L6-v2` | 384 | English | Fast prototyping, English-only |
# | `multilingual-MiniLM` | 384 | 50+ | Multilingual, runs locally ✅ |
# | `BGE-M3` | 1024 | 100+ | State-of-the-art multilingual |
# | `text-embedding-3-large` | 3072 | Multi | OpenAI API (highest quality) |
#
# ### Our Choice
# We use `paraphrase-multilingual-MiniLM-L12-v2` (an encoder-only model):
# - **384 dimensions** per embedding
# - Supports **50+ languages** (including Italian)
# - Runs **locally on CPU** — no API key needed
# - Good balance of speed and quality

# %%
from sentence_transformers import SentenceTransformer

def generate_embeddings(terms: list[dict], model_name: str = None, lang: str = "en") -> np.ndarray:
    """Generate embeddings for all terms."""
    if model_name is None:
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"

    print(f"⟳ Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    text_list = [t[lang] for t in terms]
    print(f"⟳ Generating embeddings for {len(text_list)} terms ({lang.upper()})...")
    embeddings = model.encode(text_list, show_progress_bar=True)

    print(f"✓ Generated embeddings: shape = {embeddings.shape}")
    print(f"  Each term is now a vector of {embeddings.shape[1]} dimensions")
    return embeddings

embeddings = generate_embeddings(data["terms"])

# %% [markdown]
# Let's peek at what an embedding actually looks like:

# %%
# Visualize the first embedding as a bar chart
term_0 = data["terms"][0]["en"]
vec = embeddings[0]

fig, ax = plt.subplots(figsize=(12, 2.5))
colors = ['#2196F3' if v >= 0 else '#E91E63' for v in vec[:50]]
ax.bar(range(50), vec[:50], color=colors, width=0.8)
ax.set_xlabel("Dimension index", fontsize=11)
ax.set_ylabel("Value", fontsize=11)
ax.set_title(f'Embedding for "{term_0}" (first 50 of {len(vec)} dimensions)', fontsize=13)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.set_xlim(-1, 50)
plt.tight_layout()
plt.show()

print(f'\n📊 Embedding stats for "{term_0}":')
print(f"   Shape: ({len(vec)},) · Norm: {np.linalg.norm(vec):.4f} · Min: {vec.min():.4f} · Max: {vec.max():.4f}")

# %% [markdown]
# ### Why Dimensions Matter
# From the slides: think of dimensions like **image resolution**.
# - 384 dimensions = "720p" — good enough for most tasks, fast and cheap
# - 3072 dimensions = "4K" — captures subtle differences, but slower and more expensive
#
# Our model uses 384 dimensions. Let's see what that means in practice:

# %%
n_terms, n_dims = embeddings.shape
bytes_per_float = 4
total_384 = n_terms * n_dims * bytes_per_float
total_3072 = n_terms * 3072 * bytes_per_float

df_storage = pd.DataFrame([
    {"Model": "Our model (MiniLM)", "Dims": n_dims, f"Storage ({n_terms} terms)": f"{total_384/1024:.1f} KB"},
    {"Model": "OpenAI large", "Dims": 3072, f"Storage ({n_terms} terms)": f"{total_3072/1024:.1f} KB"},
])
print(f"💾 Storage Comparison")
print(f"   → {3072//n_dims}x more storage for higher-dim models. Like 720p vs 4K.\n")
df_storage

# %% [markdown]
# ---
# ## Step 4: Compute Cosine Similarity
#
# Cosine similarity measures the **angle between two vectors**.
# From the slides:
#
# $$\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}$$
#
# > 📄 **From the slides — Distance Metrics:**
# > There are 3 ways to measure vector similarity:
# > - **Cosine**: angle between vectors (ignores magnitude) ← *we use this*
# > - **Dot Product**: magnitude-sensitive (longer vectors score higher)
# > - **Euclidean**: L2 distance (smaller = more similar)
# >
# > When vectors are normalized (ours are!), cosine = dot product.
#
# - **1.0** = identical direction → same meaning ("car" ↔ "automobile")
# - **0.0** = perpendicular → unrelated ("car" ↔ "happiness")
# - In practice, similar texts score **0.7–0.95**

# %%
def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between all embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(embeddings)

sim_matrix = compute_similarity_matrix(embeddings)
print(f"✓ Similarity matrix computed: shape = {sim_matrix.shape}")

# %% [markdown]
# ---
# ## Step 5: Visualize the Similarity Matrix
#
# We display the cosine similarity matrix as a **heatmap**.
# You should see bright blocks along the diagonal and within same-category
# groups — proving that embeddings capture semantic structure.

# %%
labels = [t["en"] for t in data["terms"]]
categories = [t["category"] for t in data["terms"]]

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(labels, fontsize=10)

# Add similarity values in each cell
for i in range(len(labels)):
    for j in range(len(labels)):
        val = sim_matrix[i, j]
        color = "white" if val > 0.65 or val < 0.2 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

# Add category color bars on the side
unique_cats = list(dict.fromkeys(categories))
cat_cmap = {"technology": "#2196F3", "finance": "#4CAF50", "sports": "#FF9800", "food": "#E91E63"}
for i, cat in enumerate(categories):
    ax.add_patch(plt.Rectangle((-1.8, i - 0.5), 0.6, 1, color=cat_cmap.get(cat, "#888"), clip_on=False))

# Legend
for cat, color in cat_cmap.items():
    ax.plot([], [], 's', color=color, label=cat, markersize=10)
ax.legend(loc='upper left', bbox_to_anchor=(1.12, 1), title="Category", fontsize=10)

plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
ax.set_title("Cosine Similarity Heatmap", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Step 6: 2D Cluster Map — See the "Meaning Space" 🗺️
#
# The heatmap shows numbers, but can we actually **see** the clusters?
# We use **PCA** (Principal Component Analysis) to compress our 384-dimensional
# embeddings down to just **2 dimensions** — like squashing a globe into a flat map.
#
# > 💡 **From the slides:** Remember the GPS analogy? This is the actual map!
# > Similar terms should appear close together, forming visible clusters.

# %%
from sklearn.decomposition import PCA

# Reduce 384 dimensions → 2 dimensions
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(12, 8))

cat_cmap = {"technology": "#2196F3", "finance": "#4CAF50", "sports": "#FF9800", "food": "#E91E63"}

# Plot each category with different color
for cat, color in cat_cmap.items():
    mask = [t["category"] == cat for t in data["terms"]]
    cat_coords = coords[mask]
    cat_labels = [t["en"] for t, m in zip(data["terms"], mask) if m]
    ax.scatter(cat_coords[:, 0], cat_coords[:, 1], c=color, s=150, label=cat,
              edgecolors='white', linewidth=1.5, zorder=3, alpha=0.9)
    # Add labels to each point
    for (x, y), label in zip(cat_coords, cat_labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 5),
                    fontsize=9, color=color, fontweight='bold')

# Style
ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=11)
ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=11)
ax.set_title("2D Embedding Map — Terms Clustered by Meaning", fontsize=14, fontweight='bold')
ax.legend(title="Category", fontsize=11, title_fontsize=12)
ax.grid(True, alpha=0.2)
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()

total_var = sum(pca.explained_variance_ratio_[:2])
print(f"🔍 What to notice:")
print(f"   • Terms in the same category cluster together")
print(f"   • Categories are separated in space")
print(f"   • Some terms are closer across categories — they share semantic overlap")
print(f"   • PCA explains only {total_var:.0%} of variance with 2 axes — the real space is 384D!")

# %% [markdown]
# ---
# ## Step 7: Find the Most Similar Pairs
#
# Now let's find which term pairs have the **highest cosine similarity**.
# If embeddings truly capture meaning, the most similar pairs should
# be terms from the **same category**.

# %%
def find_top_similar_pairs(similarity_matrix, terms, top_k=10):
    """Find and display the top-K most similar term pairs."""
    pairs = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            score = similarity_matrix[i][j]
            pairs.append((score, terms[i]["en"], terms[j]["en"],
                          terms[i]["category"], terms[j]["category"]))
    pairs.sort(reverse=True)

    rows = []
    for idx, (score, t1, t2, c1, c2) in enumerate(pairs[:top_k], 1):
        same = "✓ Yes" if c1 == c2 else "✗ No"
        rows.append({"#": idx, "Term 1": t1, "Term 2": t2,
                      "Similarity": f"{score:.3f}", "Same Category?": same})

    same_cat = sum(1 for _, _, _, c1, c2 in pairs[:top_k] if c1 == c2)

    print(f"🏆 Top {top_k} Most Similar Term Pairs\n")
    display(pd.DataFrame(rows).set_index("#"))
    print(f"\n📊 Observation: {same_cat}/{top_k} of the most similar pairs are in the same category")
    print(f"   → Embeddings capture semantic structure!")

find_top_similar_pairs(sim_matrix, data["terms"], top_k=10)

# %% [markdown]
# ---
# ## Step 8: Semantic Search — "Find Similar Meaning" 🔍
#
# This is **the core use case** that everything else in the course builds on.
# In Lecture 3 (RAG), you'll use this exact technique to search documents by meaning.
#
# We embed a **free-text query** and find the closest terms from our dataset.
# Unlike keyword search, this finds results based on **meaning**, not exact words.

# %%
from sentence_transformers import SentenceTransformer

# We already have our model loaded — let's reuse it
search_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def semantic_search(query: str, terms: list[dict], embeddings: np.ndarray, top_k: int = 5):
    """Find the most semantically similar terms to a free-text query."""
    query_emb = search_model.encode([query])
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(query_emb, embeddings)[0]

    # Sort by score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    rows = []
    for rank, (idx, score) in enumerate(ranked, 1):
        term = terms[idx]
        rows.append({"#": rank, "Match": term["en"], "Category": term["category"],
                      "Similarity": f"{score:.3f}"})

    print(f'\n🔍 Query: "{query}"\n')
    display(pd.DataFrame(rows).set_index("#"))

# %%
# Try different queries to see how semantic search works!
semantic_search("I love playing football on the weekend", data["terms"], embeddings)

# %%
semantic_search("how to invest money in the stock market", data["terms"], embeddings)

# %%
semantic_search("building a neural network with Python", data["terms"], embeddings)

# %% [markdown]
# > 💡 **Key insight:** Notice how the search finds relevant terms even though
# > **none of the exact words match**. "football" finds "soccer" and "basketball".
# > "invest money" finds "stock market" and "investment portfolio".
# > This is the power of **semantic search** — and the foundation of RAG.
#
# ### 🧪 Try it yourself!
# Type any sentence below and see what the model finds:

# %%
# ✏️ Type your own query here!
semantic_search("my grandmother makes the best pasta", data["terms"], embeddings)

# %% [markdown]
# ---
# ## Step 9: Word2Vec Analogy — The Classic Demo 👑
#
# > 📄 **From the slides — Word2Vec (Mikolov et al., Google 2013):**
# > The paper that started the embedding revolution showed that vector
# > arithmetic captures semantic relationships:
# > `king − man + woman ≈ queen`
#
# Let's try this with our model!

# %%
_analogy_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def word2vec_analogy(word_a, word_b, word_c, model, top_k=3):
    """Compute: word_a - word_b + word_c ≈ ?"""
    vecs = model.encode([word_a, word_b, word_c])
    # Perform vector arithmetic
    result_vec = vecs[0] - vecs[1] + vecs[2]
    result_vec = result_vec.reshape(1, -1)
    
    # Find closest words from a candidate list
    candidates = ["queen", "king", "man", "woman", "prince", "princess",
                  "boy", "girl", "emperor", "empress", "husband", "wife",
                  "father", "mother", "son", "daughter", "Rome", "Italy",
                  "Paris", "France", "Berlin", "Germany", "Madrid", "Spain"]
    cand_vecs = model.encode(candidates)
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity(result_vec, cand_vecs)[0]
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = " | ".join(f"{w} ({s:.3f})" for w, s in ranked)
    print(f"   {word_a} − {word_b} + {word_c} ≈ {results}")

print("👑 Word2Vec-style analogies:\n")
word2vec_analogy("king", "man", "woman", _analogy_model)
word2vec_analogy("Paris", "France", "Germany", _analogy_model)
word2vec_analogy("husband", "man", "woman", _analogy_model)

# %% [markdown]
# ### 🧪 Try it yourself!
# Create your own analogy: `A − B + C ≈ ?`

# %%
# ✏️ Change the words and re-run!
word2vec_analogy("Rome", "Italy", "France", _analogy_model)

# %%
del _analogy_model

# %% [markdown]
# ---
# ## Step 10: Model Comparison (Bonus) 🏆
#
# Compare how a **multilingual** model vs an **English-only** model handles
# the same terms in **Italian**. From the slides:
#
# > 💡 Multilingual models map different languages into a **shared vector space**.
# > "heart failure" (English) ≈ "insufficienza cardiaca" (Italian)
#
# Here we switch to embedding the **Italian versions** of our terms and compare
# how well each model clusters them. The multilingual model should produce
# **tighter clusters** (higher within-category similarity) for Italian.
#
# Uncomment the cell below to run this comparison:

# %%
def compare_models(terms: list[dict]):
    """Compare multilingual vs English-only embedding models."""
    models = [
        ("paraphrase-multilingual-MiniLM-L12-v2", "Multilingual"),
        ("all-MiniLM-L6-v2", "English-only"),
    ]

    results = {}
    for model_name, label in models:
        print(f"\n{'='*50}")
        print(f"{label} Model: {model_name}")
        print(f"{'='*50}")

        emb = generate_embeddings(terms, model_name, lang="it")
        sim = compute_similarity_matrix(emb)

        cat_avgs = {}
        categories = set(t["category"] for t in terms)
        for cat in sorted(categories):
            indices = [i for i, t in enumerate(terms) if t["category"] == cat]
            if len(indices) < 2:
                continue
            cat_sims = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    cat_sims.append(sim[indices[i]][indices[j]])
            cat_avgs[cat] = np.mean(cat_sims)
        results[label] = cat_avgs

    # Visualize as grouped bar chart
    cats = sorted(results["Multilingual"].keys())
    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, [results["Multilingual"][c] for c in cats], width, label="Multilingual", color="#2196F3")
    bars2 = ax.bar(x + width/2, [results["English-only"][c] for c in cats], width, label="English-only", color="#FF9800")

    ax.set_ylabel("Avg Within-Category Similarity")
    ax.set_title("Model Comparison: Italian Terms (higher = better clustering)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.bar_label(bars1, fmt="%.2f", fontsize=9)
    ax.bar_label(bars2, fmt="%.2f", fontsize=9)
    plt.tight_layout()
    plt.show()

# Uncomment the line below to run the bonus comparison:
# compare_models(data["terms"])

# %% [markdown]
# ---
# ## 🎯 Summary
#
# ### Key Takeaways
# - **Embeddings** capture semantic meaning as high-dimensional vectors
# - **Cosine similarity** measures how related two pieces of text are (0.0 → 1.0)
# - Terms in the **same category cluster together** — proving embeddings understand meaning
# - **Multilingual models** handle non-English text better than English-only models
#
# ### Connection to the Course
# Everything from Lecture 2 onward **builds on embeddings**:
# - **Lecture 2 (Prompting):** How to communicate with LLMs effectively
# - **Lecture 3 (RAG):** Use embeddings to search documents by meaning
# - **Lecture 5 (Advanced RAG):** Hybrid retrieval with BM25 + embeddings
# - **Lecture 9 (Fine-Tuning):** Train your own embedding model for your domain
