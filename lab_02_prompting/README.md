# Lab 02: Prompt Engineering Workshop

> **Lecture:** 3 — Prompt Engineering
> **Duration:** 35 minutes
> **Difficulty:** Beginner

## Objectives

- Understand why naive prompts fail (hallucination, verbosity, wrong format)
- Practice zero-shot, few-shot, and Chain-of-Thought prompting
- Learn to force structured JSON output from LLMs
- Compare 5 different prompt strategies on the same query
- Experiment with Temperature and Top-P to understand generation parameters
- Understand prompt injection attacks and how to defend against them

## Prerequisites

| Requirement | How to Check |
|---|---|
| Python 3.10+ | `python --version` |
| Dependencies | `pip install openai rich python-dotenv` |
| LLM access | Google AI Studio API key in `.env` as `GOOGLE_API_KEY` |

> **Get a free API key:** Visit [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — takes 30 seconds.

> **Note:** This lab does NOT require RAG or a vector database. Context is provided manually so you can focus 100% on prompting techniques. RAG is covered in the **next lab** (Lab 03).

## Quick Start

```bash
cd lab_02_prompting
python lab_02_prompting.py
```

## Files

| File | Description |
|---|---|
| `lab_02_prompting.py` | Main script — 10 steps, 5 prompt strategies compared side-by-side |
| `lab_02_prompting.ipynb` | Jupyter notebook version |
| `solutions.py` | Reference implementation for the Chain-of-Thought TODO |

## Lab Steps

| Step | Topic | What You'll See |
|---|---|---|
| 1 | Hello World | API connection test |
| 2 | Zero-Shot | Verbose, unpredictable output (the "before") |
| 3 | System Message | One-word answer, controlled output (the "after") |
| 4 | Few-Shot | Format-perfect output from examples alone |
| 5 | Chain-of-Thought | Step-by-step reasoning shown explicitly |
| 6 | Structured JSON | Production-ready parseable output |
| 7 | Grand Comparison | All 5 strategies side-by-side |
| 8 | Temperature | Deterministic → creative → chaotic |
| 8b | Top-P & max_tokens | Vocabulary pool and output length controls |
| 9 | Prompt Injection | Real attack attempts on a scoped bot |
| 9b | Scope Guard Defense | Prompt chaining as a security mechanism |
| 10 | Production Prompt | System + Few-Shot + CoT + JSON combined |

## Tasks

### Task 1: Run All Steps
Run the script end-to-end and observe how the **same review** produces different answers with each prompt strategy.

### Task 2: Observe the Temperature Effect (Step 8)
Compare outputs at T=0.0, T=0.5, T=1.0, T=1.5. At what point does the output become unusable?

### Task 3: Try a New Injection Attack (Step 9)
Modify one of the three attack strings and see if Gemini resists your custom attack.

### Task 4: Implement Chain-of-Thought (Bonus)
Find the `step_5_cot` function and try writing your own CoT prompt before looking at `solutions.py`.

## Key Concepts

| Strategy | When to Use |
|---|---|
| **Zero-Shot** | Quick experiments, not for production |
| **System Message** | Always — the single highest-return improvement |
| **Few-Shot** | When you need consistent format or tone |
| **Chain-of-Thought** | Complex reasoning; large models only |
| **Structured Output** | Any time downstream code needs to parse the answer |
| **Prompt Chaining** | Complex multi-step tasks; security (scope guard) |
