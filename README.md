# 🎓 LLM Course for Enterprise Developers

> **20 hours · 10 lectures × 2 hours · Zucchetti**
> From transformer internals to production LLM systems — with math, papers, and hands-on labs.

**Instructor:** Kamyar Zeinalipour

---

## 📚 Course Overview

A comprehensive, theory-heavy course designed for software engineers who want to understand **how** and **why** LLMs work — not just how to use them. Each lecture includes academic paper references, mathematical foundations, and a hands-on lab.

```
📘 FOUNDATIONS (Lectures 1–4)                📙 ADVANCED (Lectures 5–10)
┌──────────────────────────────┐      ┌──────────────────────────────────────┐
│ ✅ L1  Transformer Blueprint  │      │ 🔒 L5  Advanced RAG & Routing        │
│ ✅ L2  Tokenization & Embed.  │      │ 🔒 L6  Graph RAG & Knowledge Graphs  │
│ ✅ L3  Prompt Engineering     │      │ 🔒 L7  Agentic RAG & Autonomous      │
│ 🔒 L4  RAG & Decision Matrix  │      │ 🔒 L8  RAG Evaluation & Observability│
└──────────────────────────────┘      │ 🔒 L9  Fine-Tuning (LoRA / QLoRA)   │
                                       │ 🔒 L10 Multimodal & Production      │
                                       └──────────────────────────────────────┘

✅ = Available now    🔒 = Coming soon
```

---

## 📖 Lecture 1: The Transformer Blueprint

**Duration:** 2 hours

### 🎯 What You'll Learn

- **Transformer architecture** — the math behind self-attention, multi-head attention, positional encoding
- **How LLMs work** — next-token prediction, cross-entropy loss, temperature & sampling
- **Key concepts** — hallucination, perplexity, context windows, RLHF
- **Encoder vs Decoder** — the 3 architecture families and when to use each
- **Scale & emergent abilities** — why bigger models develop unexpected capabilities

### 📄 Key Papers Covered

| Year | Paper | Venue |
|------|-------|-------|
| 2013 | Word2Vec — *Efficient Estimation of Word Representations* (Mikolov et al.) | Google |
| 2014 | GloVe — *Global Vectors for Word Representation* (Pennington et al.) | EMNLP |
| 2016 | BPE — *Neural Machine Translation of Rare Words* (Sennrich et al.) | ACL |
| 2017 | *Attention Is All You Need* (Vaswani et al.) | NeurIPS |
| 2021 | RoPE — *Rotary Position Embedding* (Su et al.) | — |
| 2022 | InstructGPT/RLHF (Ouyang et al.) | NeurIPS |
| 2023 | LLaMA (Touvron et al.) | Meta |

### 📂 Materials

| Resource | Path | Description |
|----------|------|-------------|
| 🎨 **Slides** | [**▶ Open Presentation**](https://kamyarzeinalipour.github.io/zucchetti-llm-course/slides/lecture_01_llm_embeddings.html) | Click to view slides in your browser |
| 🔬 **Lab notebook** | [`lab_01_embeddings/lab_01_embeddings.py`](lab_01_embeddings/lab_01_embeddings.py) | Python script — run locally |

---

## 📖 Lecture 2: Tokenization, Embeddings & Similarity

**Duration:** ~1.5 hours

### 🎯 What You'll Learn

- **Tokenization** — BPE algorithm step-by-step, vocabulary sizes, why non-English text costs more tokens
- **Why numbers break** — how tokenization fragments numbers and why LLMs can't do arithmetic
- **Embeddings** — from Word2Vec to modern sentence transformers, GPS analogy for meaning
- **Similarity metrics** — cosine similarity, dot product, Euclidean distance, and how to interpret scores
- **Multilingual alignment** — cross-lingual search, domain-specific challenges
- **Model selection** — MiniLM vs BGE-M3 vs OpenAI, when to use each

### 📄 Key Papers Covered

| Year | Paper | Venue |
|------|-------|-------|
| 2013 | Word2Vec — *Efficient Estimation of Word Representations* (Mikolov et al.) | Google |
| 2014 | GloVe — *Global Vectors for Word Representation* (Pennington et al.) | EMNLP |
| 2016 | BPE — *Neural Machine Translation of Rare Words* (Sennrich et al.) | ACL |

### 📂 Materials

| Resource | Path | Description |
|----------|------|-------------|
| 🎨 **Slides** | [**▶ Open Presentation**](https://kamyarzeinalipour.github.io/zucchetti-llm-course/slides/lecture_02_tokenization_embeddings.html) | Click to view slides in your browser |
| 🔬 **Lab notebook** | [`lab_01_embeddings/lab_01_embeddings.py`](lab_01_embeddings/lab_01_embeddings.py) | Shared lab — embeddings exploration |

---

## 📖 Lecture 3: Prompt Engineering

**Duration:** 2 hours

### 🎯 What You'll Learn

- **Zero-shot, few-shot, chain-of-thought** — five prompting strategies from basic to advanced
- **Structured output (JSON)** — how to make LLMs production-ready
- **Temperature, top-p, top-k** — the math behind generation control
- **Prompt injection** — security attacks and defenses
- **The language effect** — why English prompts work better even for Italian tasks

### 📄 Key Papers Covered

| Year | Paper | Venue |
|------|-------|-------|
| 2020 | GPT-3 — *Language Models are Few-Shot Learners* (Brown et al.) | NeurIPS |
| 2022 | *Chain-of-Thought Prompting Elicits Reasoning* (Wei et al.) | NeurIPS |
| 2022 | *Rethinking the Role of Demonstrations* (Min et al.) | EMNLP |
| 2023 | *Self-Consistency Improves CoT Reasoning* (Wang et al.) | ICLR |

### 📂 Materials

| Resource | Path | Description |
|----------|------|-------------|
| 🎨 **Slides** | [**▶ Open Presentation**](https://kamyarzeinalipour.github.io/zucchetti-llm-course/slides/lecture_03_prompt_engineering.html) | Click to view slides in your browser |
| 🔬 **Lab notebook** | [`lab_03_prompting/lab_03_prompting.py`](lab_03_prompting/lab_03_prompting.py) | Prompt engineering workshop |

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **Google AI Studio API key** (free) — [Get one here](https://aistudio.google.com/apikey)

### 2. Clone & Install

```bash
git clone https://github.com/KamyarZeinalipour/zucchetti-llm-course.git
cd zucchetti-llm-course

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your Google AI Studio API key
```

### 4. Run Lab 1

```bash
cd lab_01_embeddings
python lab_01_embeddings.py
```

### 5. View Slides

Open `slides/lecture_01_llm_embeddings.html` in your browser.

---

## 📅 Release Schedule

| Lecture | Topic | Status |
|---------|-------|--------|
| Lecture 1 | The Transformer Blueprint | ✅ **Available** |
| Lecture 2 | Tokenization, Embeddings & Similarity | ✅ **Available** |
| Lecture 3 | Prompt Engineering | ✅ **Available** |
| Lecture 4 | RAG Foundations & Decision Matrix | 🔒 Coming soon |
| Lecture 5 | Advanced RAG & Intelligent Routing | 🔒 Coming soon |
| Lecture 6 | Graph RAG & Knowledge Graphs | 🔒 Coming soon |
| Lecture 7 | Agentic RAG & Autonomous Systems | 🔒 Coming soon |
| Lecture 8 | RAG Evaluation & Observability | 🔒 Coming soon |
| Lecture 9 | Efficient Fine-Tuning (LoRA / QLoRA) | 🔒 Coming soon |
| Lecture 10 | Multimodal AI & Production Engineering | 🔒 Coming soon |

Materials for each lecture will be released before the corresponding class session.

---

## 📁 Repository Structure

```
zucchetti-llm-course/
├── README.md                        ← You are here
├── requirements.txt                 ← Python dependencies
├── .env.example                     ← API key template
│
├── slides/                          ← Lecture presentations
│   ├── lecture_01_llm_embeddings.html        ← L1 slides
│   ├── lecture_02_tokenization_embeddings.html ← L2 slides
│   └── img_*.png / img_*.svg                ← Slide images
│
└── lab_01_embeddings/               ← Hands-on lab (L1 & L2)
    ├── lab_01_embeddings.py             ← Main script
    ├── lab_01_embeddings.ipynb          ← Jupyter notebook
    ├── solutions.py                     ← Complete solutions
    └── data/                            ← Sample datasets
```

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| API rate limit (429) | Wait 30 seconds and retry — free tier is 15 req/min |
| Slides look broken | Use Chrome or Firefox — Safari may have rendering issues |
| API key not working | Check [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

---

## 📜 License

This course material is provided for educational purposes at Zucchetti S.p.A. All datasets are subsets of publicly available sources.
