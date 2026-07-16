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
# ⏱ **Duration:** ~20 minutes · 🔒 **No API key needed** — runs 100% locally

# %% [markdown]
# ---
# ## ☁️ Colab Setup
# If you're running this on **Google Colab**, run the cell below to
# clone the course repository and install dependencies.
# If you're running locally, skip to "Setup & Imports".

# %%
# --- Run this cell in Google Colab ---
!pip install -q sentence-transformers transformers numpy scikit-learn matplotlib

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
from pathlib import Path
from IPython.display import display, HTML
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
# > We'll see this effect hands-on below and again in the bonus Step 7.

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

# Build HTML table for the dataset
cat_colors = {"technology": "#2196F3", "finance": "#4CAF50", "sports": "#FF9800", "food": "#E91E63"}
rows_html = ""
for i, t in enumerate(data["terms"], 1):
    cat = t["category"]
    badge = f'<span style="background:{cat_colors.get(cat,"#888")}; color:white; padding:2px 8px; border-radius:10px; font-size:12px;">{cat}</span>'
    rows_html += f"<tr><td>{i}</td><td><strong>{t['en']}</strong></td><td style='color:#888;'>{t['it']}</td><td>{badge}</td></tr>"

display(HTML(f"""
<div style="font-family:sans-serif;">
<h3>📋 Terms Dataset — {len(data['terms'])} terms across {len(cat_colors)} categories</h3>
<table style="border-collapse:collapse; width:100%; max-width:650px;">
<tr style="background:#f5f5f5; border-bottom:2px solid #ddd;">
  <th style="padding:8px; text-align:left;">#</th>
  <th style="padding:8px; text-align:left;">English</th>
  <th style="padding:8px; text-align:left;">Italian</th>
  <th style="padding:8px; text-align:left;">Category</th>
</tr>
{rows_html}
</table>
</div>
"""))

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

TOKEN_COLORS = [
    "#a8d8ea", "#aa96da", "#fcbad3", "#ffffd2", "#b5eaea",
    "#f6c6c6", "#c9e4de", "#d4a5a5", "#e8d5b7", "#b8e0d2",
]

def render_tokens_html(text, tokenizer):
    """Render tokenized text as colored HTML boxes."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    spans = []
    for i, tok in enumerate(tokens):
        color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
        display_tok = tok.replace("##", "·")
        spans.append(
            f'<span style="background:{color}; padding:3px 6px; margin:2px; '
            f'border-radius:4px; font-family:monospace; font-size:14px; '
            f'display:inline-block;">{display_tok}</span>'
        )
    return "".join(spans)

examples = [
    ("machine learning",           "apprendimento automatico"),
    ("artificial intelligence",    "intelligenza artificiale"),
    ("investment portfolio",       "portafoglio di investimenti"),
]

html_parts = [
    '<div style="font-family:sans-serif; max-width:700px;">',
    '<h3>🔍 How the Tokenizer Splits Text</h3>',
]

for en, it in examples:
    en_tokens = tokenizer.encode(en, add_special_tokens=False)
    it_tokens = tokenizer.encode(it, add_special_tokens=False)
    html_parts.append(
        f'<div style="margin:12px 0; padding:12px; background:#f8f8f8; border-radius:8px;">'
        f'<div style="margin-bottom:8px;">'
        f'<strong style="color:#2196F3;">EN</strong> "{en}" → '
        f'<strong>{len(en_tokens)} tokens</strong></div>'
        f'<div style="margin-bottom:10px;">{render_tokens_html(en, tokenizer)}</div>'
        f'<div style="margin-bottom:8px;">'
        f'<strong style="color:#FF9800;">IT</strong> "{it}" → '
        f'<strong>{len(it_tokens)} tokens</strong> '
        f'<span style="color:red;">(+{len(it_tokens) - len(en_tokens)})</span></div>'
        f'<div>{render_tokens_html(it, tokenizer)}</div>'
        f'</div>'
    )

html_parts.append(
    '<p style="color:#888; font-size:13px; margin-top:8px;">'
    '· = subword boundary (the tokenizer split a word into pieces)<br>'
    'More tokens = higher API cost + less fits in the context window</p>'
    '</div>'
)

display(HTML("".join(html_parts)))

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
from transformers import pipeline

# --- 1. DECODER-ONLY: Text Generation (like ChatGPT) ---
print("⟳ Loading decoder-only model (GPT-2)...")
generator = pipeline("text-generation", model="gpt2", max_new_tokens=30, do_sample=True, temperature=0.7)
prompt = "Artificial intelligence will"
result = generator(prompt)[0]["generated_text"]

# --- 2. ENCODER-DECODER: Text Transformation (like T5) ---
# We'll show BOTH parts: what the encoder produces AND what the decoder generates
print("⟳ Loading encoder-decoder model (T5)...")
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

en_text = "translate English to German: Machine learning is a subset of artificial intelligence."
t5_inputs = t5_tokenizer(en_text, return_tensors="pt")

# Step A: Run the ENCODER → get hidden states (the intermediate representation)
with torch.no_grad():
    encoder_outputs = t5_model.encoder(**t5_inputs)
encoder_hidden = encoder_outputs.last_hidden_state  # shape: (1, seq_len, 512)

# Step B: Run the DECODER → generate translation from those hidden states
output_ids = t5_model.generate(**t5_inputs, max_length=50)
translation = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Format encoder hidden states for display
enc_shape = encoder_hidden.shape
enc_vals = encoder_hidden[0][0][:5].tolist()  # first token, first 5 dims
enc_vals_str = ", ".join(f"{v:.4f}" for v in enc_vals)

# --- 3. ENCODER-ONLY: Embeddings (what we use!) ---
from sentence_transformers import SentenceTransformer
print("⟳ Loading encoder-only model (MiniLM)...")
_demo_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
_demo_emb = _demo_model.encode(["machine learning"])

# Display all 3 results
display(HTML(f"""
<div style="font-family:sans-serif; max-width:800px;">
<h3>🔬 Three Transformer Architectures in Action</h3>

<div style="margin:10px 0; padding:14px; background:#fff3e0; border-left:4px solid #FF9800; border-radius:4px;">
  <strong style="color:#FF9800;">1. Decoder-only</strong> (GPT-2) — <em>Generates text token by token</em><br>
  <code style="background:#f5f5f5; padding:2px 6px; border-radius:3px;">Input: "{prompt}"</code><br>
  <div style="margin-top:6px; padding:8px; background:#f5f5f5; border-radius:4px; font-family:monospace;">{result}</div>
  <div style="color:#888; font-size:12px; margin-top:4px;">→ This is how ChatGPT, LLaMA, and Mistral work</div>
</div>

<div style="margin:10px 0; padding:14px; background:#e8f5e9; border-left:4px solid #4CAF50; border-radius:4px;">
  <strong style="color:#4CAF50;">2. Encoder-Decoder</strong> (T5) — <em>Two-stage: encode then decode</em><br>
  <code style="background:#f5f5f5; padding:2px 6px; border-radius:3px;">Input: "Machine learning is a subset of artificial intelligence."</code>

  <div style="margin:10px 0; padding:10px; background:#c8e6c9; border-radius:4px;">
    <strong>Stage 1 — ENCODER</strong> (understands the input):<br>
    <div style="font-family:monospace; font-size:12px; margin-top:4px;">
      Input tokens: {enc_shape[1]} tokens → Hidden states: <strong>{enc_shape[1]} × {enc_shape[2]} dimensions</strong><br>
      Each token gets a rich representation: [{enc_vals_str}, ...]
    </div>
    <div style="color:#2e7d32; font-size:11px; margin-top:2px;">↓ These hidden states are passed to the decoder ↓</div>
  </div>

  <div style="margin:10px 0; padding:10px; background:#c8e6c9; border-radius:4px;">
    <strong>Stage 2 — DECODER</strong> (generates the output from hidden states):<br>
    <div style="font-family:monospace; font-size:13px; margin-top:4px;">
      Output: <strong>{translation}</strong>
    </div>
  </div>
  <div style="color:#888; font-size:12px;">→ Used for translation, summarization, Q&A</div>
</div>

<div style="margin:10px 0; padding:14px; background:#e3f2fd; border-left:4px solid #2196F3; border-radius:4px;">
  <strong style="color:#2196F3;">3. Encoder-only</strong> (MiniLM) — <em>Understands text → produces embeddings</em><br>
  <code style="background:#f5f5f5; padding:2px 6px; border-radius:3px;">Input: "machine learning"</code><br>
  <div style="margin-top:6px; padding:8px; background:#f5f5f5; border-radius:4px; font-family:monospace;">Output: [{_demo_emb[0][0]:.4f}, {_demo_emb[0][1]:.4f}, {_demo_emb[0][2]:.4f}, ...] ({_demo_emb.shape[1]} dims)</div>
  <div style="color:#888; font-size:12px; margin-top:4px;">→ This is what we use in this lab! ✅</div>
</div>

<div style="margin-top:12px; padding:10px; background:#fafafa; border-radius:8px; border:1px solid #ddd;">
  <strong>💡 Key insight:</strong> The encoder-decoder's <em>encoder hidden states</em> are similar to embeddings!
  The difference is that encoder-decoder feeds them to a decoder for generation,
  while encoder-only models return them directly for tasks like search and similarity.
</div>
</div>
"""))

del generator, t5_model, t5_tokenizer, _demo_model, _demo_emb  # free memory


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

display(HTML(f"""
<div style="font-family:sans-serif; background:#f8f8f8; padding:12px; border-radius:8px; max-width:500px; margin-top:8px;">
  <strong>📊 Embedding stats for "{term_0}":</strong><br>
  <code>Shape: ({len(vec)},) · Norm: {np.linalg.norm(vec):.4f} · Min: {vec.min():.4f} · Max: {vec.max():.4f}</code>
</div>
"""))

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

display(HTML(f"""
<div style="font-family:sans-serif; max-width:500px;">
<h4>💾 Storage Comparison</h4>
<table style="border-collapse:collapse; width:100%;">
<tr style="background:#f5f5f5;"><th style="padding:8px; text-align:left;">Model</th><th style="padding:8px;">Dims</th><th style="padding:8px;">Storage ({n_terms} terms)</th></tr>
<tr><td style="padding:8px;"><strong>Our model</strong> (MiniLM)</td><td style="padding:8px; text-align:center;">{n_dims}</td><td style="padding:8px; text-align:center; color:#4CAF50;">{total_384/1024:.1f} KB</td></tr>
<tr><td style="padding:8px;">OpenAI large</td><td style="padding:8px; text-align:center;">3072</td><td style="padding:8px; text-align:center; color:#E91E63;">{total_3072/1024:.1f} KB</td></tr>
</table>
<p style="color:#888; font-size:13px; margin-top:8px;">→ {3072//n_dims}x more storage for higher-dim models. Like 720p vs 4K — more detail, but more cost.</p>
</div>
"""))

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
# ## Step 7: 2D Cluster Map — See the "Meaning Space" 🗺️
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

display(HTML("""
<div style="font-family:sans-serif; background:#e3f2fd; padding:12px; border-radius:8px; max-width:700px; margin-top:8px;">
  <strong>🔍 What to notice:</strong><br>
  • Terms in the <strong>same category</strong> cluster together<br>
  • Categories are <strong>separated</strong> in space<br>
  • Some terms are closer across categories (e.g., "artificial intelligence" near "deep learning") — these share semantic overlap<br>
  • PCA explains only ~{:.0%} of variance with 2 axes — the real space is 384D!
</div>
""".format(sum(pca.explained_variance_ratio_[:2]))))

# %% [markdown]
# ---
# ## Step 8: Find the Most Similar Pairs
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

    # Build HTML table
    rows = ""
    for idx, (score, t1, t2, c1, c2) in enumerate(pairs[:top_k], 1):
        same = c1 == c2
        badge = '<span style="color:#4CAF50; font-weight:bold;">✓ Yes</span>' if same else '<span style="color:#E91E63;">✗ No</span>'
        bar_color = "#4CAF50" if score > 0.6 else "#FF9800" if score > 0.4 else "#E91E63"
        bar_width = int(score * 100)
        bar = f'<div style="background:#eee; border-radius:4px; overflow:hidden; width:80px; display:inline-block; vertical-align:middle;"><div style="background:{bar_color}; height:14px; width:{bar_width}%;"></div></div> {score:.3f}'
        rows += f"<tr><td style='padding:6px;'>{idx}</td><td style='padding:6px;'><strong>{t1}</strong></td><td style='padding:6px;'><strong>{t2}</strong></td><td style='padding:6px;'>{bar}</td><td style='padding:6px; text-align:center;'>{badge}</td></tr>"

    same_cat = sum(1 for _, _, _, c1, c2 in pairs[:top_k] if c1 == c2)

    display(HTML(f"""
    <div style="font-family:sans-serif;">
    <h3>🏆 Top {top_k} Most Similar Term Pairs</h3>
    <table style="border-collapse:collapse; width:100%; max-width:750px;">
    <tr style="background:#f5f5f5; border-bottom:2px solid #ddd;">
      <th style="padding:8px;">#</th><th style="padding:8px; text-align:left;">Term 1</th>
      <th style="padding:8px; text-align:left;">Term 2</th><th style="padding:8px;">Similarity</th>
      <th style="padding:8px;">Same Category?</th>
    </tr>
    {rows}
    </table>
    <div style="margin-top:12px; padding:10px; background:#e8f5e9; border-radius:8px; max-width:750px;">
      <strong>📊 Observation:</strong> {same_cat}/{top_k} of the most similar pairs are in the same category
      — embeddings capture semantic structure!
    </div>
    </div>
    """))

find_top_similar_pairs(sim_matrix, data["terms"], top_k=10)

# %% [markdown]
# ---
# ## Step 9: Semantic Search — "Find Similar Meaning" 🔍
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

    # Build HTML result
    rows = ""
    for rank, (idx, score) in enumerate(ranked, 1):
        term = terms[idx]
        cat = term['category']
        cat_color = cat_cmap.get(cat, '#888')
        bar_width = int(score * 100)
        bar_color = "#4CAF50" if score > 0.5 else "#FF9800" if score > 0.3 else "#E91E63"
        bar = f'<div style="background:#eee; border-radius:4px; overflow:hidden; width:120px; display:inline-block; vertical-align:middle;"><div style="background:{bar_color}; height:16px; width:{bar_width}%;"></div></div> <strong>{score:.3f}</strong>'
        badge = f'<span style="background:{cat_color}; color:white; padding:2px 8px; border-radius:10px; font-size:11px;">{cat}</span>'
        rows += f"<tr><td style='padding:8px;'>{rank}</td><td style='padding:8px;'><strong>{term['en']}</strong></td><td style='padding:8px;'>{badge}</td><td style='padding:8px;'>{bar}</td></tr>"

    display(HTML(f"""
    <div style="font-family:sans-serif; max-width:650px;">
    <div style="background:#f3e5f5; padding:12px; border-radius:8px; margin-bottom:12px;">
      <strong>🔍 Query:</strong> "{query}"
    </div>
    <table style="border-collapse:collapse; width:100%;">
    <tr style="background:#f5f5f5; border-bottom:2px solid #ddd;">
      <th style="padding:8px;">#</th><th style="padding:8px; text-align:left;">Match</th>
      <th style="padding:8px;">Category</th><th style="padding:8px;">Similarity</th>
    </tr>
    {rows}
    </table>
    </div>
    """))

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

# %% [markdown]
# ---
# ## Step 10: Word2Vec Analogy — The Classic Demo 👑
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
    
    result_str = " | ".join(f"<strong>{w}</strong> ({s:.3f})" for w, s in ranked)
    display(HTML(f"""
    <div style="font-family:sans-serif; padding:12px; background:#f3e5f5; border-radius:8px; max-width:600px; margin:6px 0;">
      <code style="font-size:1.1em;">{word_a} − {word_b} + {word_c} ≈ ?</code><br>
      <div style="margin-top:8px;">→ {result_str}</div>
    </div>
    """))

print("👑 Word2Vec-style analogies:")
word2vec_analogy("king", "man", "woman", _analogy_model)
word2vec_analogy("Paris", "France", "Germany", _analogy_model)
word2vec_analogy("husband", "man", "woman", _analogy_model)

del _analogy_model

# %% [markdown]
# ---
# ## Step 11: Model Comparison (Bonus) 🏆
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
