# Lab 02: Prompt Engineering Workshop

> **Lecture:** 2 — Prompt Engineering
> **Duration:** 35 minutes
> **Difficulty:** Beginner

## Objectives

- Understand why naive prompts fail (hallucination, verbosity, wrong format)
- Practice zero-shot, few-shot, and Chain-of-Thought prompting
- Learn to force structured JSON output from LLMs
- Compare 5 different prompt strategies on the same query
- Build intuition for prompt design through hands-on iteration

## Prerequisites

| Requirement | How to Check |
|---|---|
| Python 3.10+ | `python --version` |
| Dependencies | `pip install openai rich python-dotenv` |
| LLM access | Ollama running locally OR DeepSeek API key in `.env` |

> **Note:** This lab does NOT require RAG or a vector database. Context is provided manually so you can focus 100% on prompting techniques. RAG is covered in the **next lab** (Lab 03).

## Quick Start

```bash
cd lab_03_prompting
python lab_03_prompting.py
```

## Files

| File | Description |
|---|---|
| `lab_03_prompting.py` | Main script — 5 prompt strategies compared side-by-side |
| `lab_03_prompting.ipynb` | Jupyter notebook version |
| `solutions.py` | Solution for the Chain-of-Thought TODO |

## Tasks

### Task 1: Run the Comparison
Run the script and observe how the **same context** produces different answers with each prompt strategy.

### Task 2: Implement Chain-of-Thought (TODO)
Find the `prompt_cot()` function and implement a CoT prompt that forces step-by-step reasoning.

### Task 3: Interactive Exploration
Use the interactive mode to test different strategies on your own questions:
- Simple factual questions (e.g., "How many remote work days?")
- Complex reasoning questions (e.g., "Compare two policies")
- Unanswerable questions (e.g., "What is the CEO's favorite color?")

## Key Concepts

| Strategy | When to Use |
|---|---|
| **Naive** | Never in production — only as a baseline |
| **Zero-Shot** | Simple, well-defined tasks |
| **Few-Shot** | When you need consistent format/tone |
| **Chain-of-Thought** | Complex reasoning, multi-step questions |
| **Structured Output** | When downstream code needs to parse the answer |
