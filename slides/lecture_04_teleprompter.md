# 🎤 Lecture 4 — Teleprompter
## RAG Foundations & Decision Matrix · Zucchetti LLM Course

> **How to use this file:**
> - Read the **bold lines** out loud — those are your key sentences.
> - *Italic text* = stage directions (don't say these out loud).
> - 🗣️ = stop and ask the room a question.
> - 💡 = extra context you can add if time allows.
> - 📄 = paper reference to mention out loud.
> - ⚠️ = common student mistake — address it directly.

---

## ⏱ Timing at a Glance

| Clock | Section | Duration |
|---|---|---|
| 0:00 | Title + Intro | 2 min |
| 0:02 | Agenda | 2 min |
| 0:04 | §1 The Retrieval Problem & RAG | 20 min |
| 0:24 | ☕ Break | 10 min |
| 0:34 | §2 Evolution of Search (table) | 3 min |
| 0:37 | §2 Boolean Search (new slide) | 3 min |
| 0:40 | §2 TF-IDF | 4 min |
| 0:44 | §2 BM25 | 4 min |
| 0:48 | §2 Dense Retrieval | 4 min |
| 0:52 | §2 Bi-Encoder vs Cross-Encoder | 3 min |
| 0:55 | §2 Sentence Transformers | 2 min |
| 0:57 | §2 Hybrid Retrieval | 2 min |
| 0:59 | §3 Document Processing & Chunking | 15 min |
| 1:14 | §4 Vector Databases & Pipeline | 10 min |
| 1:24 | §5 Decision Matrix | 10 min |
| 1:34 | Takeaways + Q&A | 5 min |
| **1:39** | **End** | |

---

## 🎬 SLIDE 1 — Title · *0:00 · 2 min*

Welcome back everyone. You just finished the lab — nice work.

Today's lecture is called **RAG Foundations and Decision Matrix.** RAG stands for Retrieval-Augmented Generation. By the end of this 90 minutes you will understand exactly how systems like ChatGPT can be connected to your own company's documents — and why that's such a powerful idea.

No code today. Pure theory. But every concept we cover today, you will implement in the next lab session — so pay attention to the *why*, not just the *what*.

*[Pause. Let people settle in.]*

---

## 🎬 SLIDE 2 — Agenda · *0:02 · 2 min*

Here is the plan for today.

*[Point to each item as you say it]*

We have five sections. We start with the big question — why do LLMs need external knowledge at all? Then we go into how search evolved over 50 years, from keywords to meaning. Then we look at how to prepare documents for retrieval — the chunking problem. Then vector databases — how you search millions of documents in milliseconds. We take a short break after section one. And we finish with the most practical slide in the whole course: the decision matrix — when should you use prompting, when RAG, when fine-tuning?

The lab is not today — it's Lecture 5. So today is your chance to build the mental model. The lab will make much more sense if this is clear.

🗣️ Before we start — how many of you have already heard the word RAG before? *[Wait for hands. If most have, say: "Good, today we're going to make sure you really understand what's happening under the hood." If few have, say: "Perfect, we're starting from scratch together."]*

---

---

# ═══ SECTION 1: THE RETRIEVAL PROBLEM & RAG ARCHITECTURE ═══
## *20 minutes total · Clock: 0:04 → 0:24*

---

## 🎬 SLIDE 3 — Section 1 Title Card · *0:04 · 1 min*

*[Section title card appears: "The Retrieval Problem — Why LLMs need external knowledge"]*

Section one. The Retrieval Problem.

Before I explain what RAG is, I need to explain *why it exists.* Because if you don't feel the problem deeply, the solution seems over-complicated. Once you feel the pain, the solution is obvious.

---

## 🎬 SLIDE 4 — The Fundamental Problem · *0:05 · 4 min*

Every large language model — GPT-4, Gemini, Claude, LLaMA — has the same two problems.

*[Point to the left card]*

**Problem one: frozen knowledge.** When a model is trained, it reads billions of pages of text from the internet. Then training stops. After that, the model's knowledge is completely frozen. It's like taking a photograph of all human knowledge on one specific day. Everything that happened after that day? The model has no idea.

GPT-4's training cut off in April 2023. It genuinely does not know what happened in May 2023. It has never seen your company's internal documents. It has never read your HR policy. It has no idea what your product does specifically.

*[Point to the right card]*

**Problem two: hallucination.** This is the dangerous one. When a model doesn't know something, most of the time it doesn't say "I don't know." It says something plausible. Something that sounds right. Something completely made up — but delivered with full confidence.

💡 A real analogy: imagine you ask a very confident colleague a question they don't know the answer to. Instead of saying "I don't know, let me check," they make something up and tell you it with complete confidence. That's hallucination. The model is not lying — it's doing exactly what it was trained to do, which is generate the most probable next word. Sometimes the most probable next word is wrong.

**The key insight in this slide: you cannot fix hallucination by just asking the model to be more careful. You fix it by giving the model real information to read.**

🗣️ Has anyone here been burned by an AI hallucinating something important? *[Give a few seconds. This usually gets hands up.]*

---

## 🎬 SLIDE 5 — Two Ways to Add Knowledge · *0:09 · 3 min*

So we know the model doesn't know your data. What do we do about it?

There are two main approaches.

*[Point to the left card]*

**Approach one: fine-tuning.** You take the model and train it further on your data. You bake the knowledge into the weights. The model literally memorizes your documents.

The problem is: this is expensive, slow, and once it's done, the knowledge is frozen again. If your documents change next month, you need to retrain. And even after fine-tuning, the model can still hallucinate — it might misremember something it "learned."

It's like memorizing a textbook before an exam. You might still remember something wrong.

*[Point to the right card — the highlighted one]*

**Approach two: RAG.** Instead of memorizing, you give the model an open book at the moment it needs to answer. You retrieve the relevant pages at query time, and you put them directly into the prompt. The model reads them and answers based on what it just read.

It's like an open-book exam. You don't need to memorize everything — you can look it up when you need it.

**Fine-tuning changes what the model knows. RAG changes what the model can read. They solve different problems.**

---

## 🎬 SLIDE 6 — The RAG Pipeline · *0:12 · 4 min*

Let me show you exactly how RAG works. This is the most important diagram of the whole lecture. It has two completely separate phases, and understanding the separation is everything.

*[Point to the top row — the offline phase]*

**Phase one: indexing — this happens offline, once.** Before any user ever asks a question, you do this setup work. You take your documents — HR manuals, product specs, any files. You split them into chunks. You feed each chunk through an embedding model and get a vector. You store all those vectors in a vector database. This is like building a very smart search index.

Why offline? Because this step is expensive. Embedding 10,000 chunks might take several seconds. You don't want to do that every time a user asks a question. You do it once, store the results, and reuse them forever.

*[Point to the bottom row — the online phase]*

**Phase two: retrieval and generation — this happens at query time, in milliseconds.** The user types a question. You embed that one question — just one embedding call, very fast. You search the database for the chunks whose vectors are most similar to the question vector. You take the top 3 or 5 chunks and put them directly into the LLM prompt. The LLM reads the chunks and generates an answer.

The whole online phase — embed the query, search the database, generate the answer — takes roughly 1 to 2 seconds total. Fast enough for any product.

💡 **Why does this architecture work so well?** Because LLMs are incredible at reading comprehension. You give them a paragraph of text and ask a question about it — they are nearly perfect at extracting the right answer. What they are *terrible* at is remembering facts from their training. RAG converts the hard problem (recall from memory) into the easy problem (reading comprehension from a given text). That's the genius of it.

📄 **Lewis et al., NeurIPS 2020** — the paper that formally introduced this architecture. The formula P(y|x) = Σ P(d|x)·P(y|x,d) says: the probability of the answer y given the question x is the sum over all documents d of: how likely we are to retrieve document d, times how likely the answer is given both the question and that document. In plain English: we try multiple retrieved documents, weight each one by how relevant it is, and combine their contributions.

---

## 🎬 SLIDE 7 — RAG in Action · *0:16 · 2 min*

Let me make this concrete with a real example.

*[Walk through the three cards slowly]*

A user asks: "What is the maternity leave policy?"

Step one — retrieve. The system searches the vector database and finds three chunks from hr_policy.pdf. Chunk one (page 12): employees are entitled to 5 months mandatory maternity leave. Chunk two (page 13): optional parental leave can extend 6 more months. Chunk three (page 14): during leave, salary is 80%.

Step two — augment and generate. Those three chunks go into the prompt. The LLM reads them. It produces: "Maternity leave is 5 months mandatory at 80% salary, with option to extend up to 6 additional months. Source: hr_policy.pdf, pages 12–14."

**The user can actually go and verify this. They can open page 12 and check. That's the power of RAG — the answer is traceable.**

📄 **Lewis et al., NeurIPS 2020** — cited on this slide.

---

## 🎬 SLIDE 8 — The Grounding Principle · *0:18 · 2 min*

Here is the before and after. This is the slide that usually makes people say "oh, I get it now."

*[Point left — the red card]*

Without RAG. User asks about the remote work policy. The model has never seen your policy. So it generates the most plausible answer: "Employees may work remotely 3 days per week." Sounds totally reasonable. Completely invented.

*[Point right — the green card]*

With RAG. Same question. The system finds the actual policy document. The model reads it and says: "According to hr_policy_2024.pdf page 8, employees may request up to 2 remote days per week after 6 months of employment." That's a real fact from a real document.

**The model is not smarter in the second case. It just has real information to read.**

⚠️ Common mistake students make: they think RAG "fixes" the model. It doesn't. It gives the model access to the right information. The model's reasoning ability is the same — but now it's reasoning about real facts instead of making things up.

💡 The slide also says: "LLMs excel at reading comprehension — RAG converts recall into comprehension." This is the core of why RAG works so well.

---

## 🎬 SLIDE 9 — RAG vs Fine-Tuning Table · *0:20 · 4 min*

Before we move on, let's compare these two approaches properly.

*[Walk through the table row by row]*

**What it changes.** RAG changes what knowledge is available at runtime. Fine-tuning changes the model's actual weights — its behavior, style, and embedded knowledge.

**Update speed.** This is where RAG wins clearly. Need to update your HR policy? With RAG, you update the document and re-index those chunks. For a typical company knowledge base (hundreds to thousands of documents) this takes minutes. For truly massive corpora (millions of documents) it could take longer. Either way: massively faster than retraining. With fine-tuning? You need to retrain. Hours to days.

**Cost.** RAG: you need embeddings and a vector database — very cheap. Fine-tuning: you need GPUs. Expensive.

**Hallucination risk.** Here is an important nuance the slide now reflects. RAG *significantly reduces* hallucination — it doesn't eliminate it. Two remaining risks: (1) the model might ignore the context and use its own training knowledge anyway, especially if the grounding instruction in the prompt is weak; (2) the model can hallucinate even when reading real text, if the relevant chunk wasn't retrieved or if multiple chunks contradict each other. The correct framing is: RAG reduces hallucination risk substantially, from "almost certain for unknown facts" to "much lower but nonzero." Always combine RAG with strong grounding prompts.

**Interpretability.** RAG: high — you can show the user which document the answer came from. Fine-tuning: low — the answer comes from weights you can't easily inspect.

**Best for.** RAG: facts, policies, Q&A. Fine-tuning: when you need a specific *style*, *format*, or *behavior* — not new facts.

💡 The most important thing to remember: **these are complementary tools, not competing ones.** Many production systems use both. You fine-tune the model to understand your company's tone and terminology, then use RAG to give it access to current documents. They work together.

*[Point to the takeaway on the slide: "RAG and fine-tuning are complementary, not competing. Many production systems use both."]*

---

## 🎬 BREAK SLIDE — ☕ 10-Minute Break · *0:24 · 10 min*

Okay, we've done the first section. You now know what the problem is and what RAG is trying to solve.

Take 10 minutes. Stand up, stretch, grab water or coffee.

When we come back we go into the technical heart of RAG — how retrieval actually works. Keywords, vectors, the math behind search. It's one of the most interesting parts.

*[Step away from the screen. Come back in exactly 10 minutes.]*

---

---

# ═══ SECTION 2: FROM KEYWORDS TO VECTORS ═══
## *20 minutes total · Clock: 0:34 → 0:54*

---

## 🎬 SLIDE 10 — Section 2 Intro · *0:34 · 1 min*

Welcome back. Section two.

This section answers a question most people never think about: how do you actually *find* the right documents when someone asks a question? The answer evolved dramatically over 50 years. Understanding that evolution will help you make better choices in your own systems.

---

## 🎬 SLIDE 11 — The Evolution of Search · *0:35 · 3 min*

Let me take you through 50 years of information retrieval in 3 minutes.

*[Walk through the table from top to bottom]*

**1970s: Boolean search.** You type "AND maternity AND leave AND policy." The system finds documents that contain all those exact words. Simple. Works. But if the document says "parental" instead of "maternity," you get zero results. No ranking. Either a document matches or it doesn't.

**1990s: TF-IDF.** Now we have ranking. Documents are scored by how often your query words appear, weighted by how rare those words are. Better. But still — if you search "time off" and the document says "vacation days," you get nothing. Still keyword-matching.

**1994+: BM25 (Okapi BM25).** Improved TF-IDF — adds saturation and length normalization. Introduced by Robertson et al. at TREC-3 in 1994, hence the name "Best Matching 25" — it was the 25th iteration. This is what Google, Elasticsearch, and Solr used for many years. Still keyword-based.

**2019 onwards: Dense retrieval.** This is the revolution. We stop matching words and start matching *meaning*. This is where your Lectures 1 and 2 knowledge about embeddings connects directly to search.

**The takeaway — and this is important:** this table is the *history* of retrieval. It does NOT mean RAG only uses dense retrieval. RAG is retrieval-method-agnostic. A simple RAG system might use just BM25 — and that's completely valid. A production system typically uses hybrid: BM25 plus dense, combined. Dense retrieval is the modern standard and gives you semantic understanding, but the "R" in RAG just means "retrieval" — it doesn't mandate which technique you use.

🗣️ **Ask the room:** "Does anyone here use Elasticsearch for search at work?" *[Hands usually go up]* "Elasticsearch uses BM25 by default. If you built a RAG system on top of Elasticsearch right now, without adding embeddings, you'd have a keyword-based RAG. It works. It's just less powerful than hybrid."

---

## 🎬 SLIDE 12 — Boolean Search · *0:38 · 3 min*

Let's spend a moment on Boolean search because most of you have probably used it without realizing it — SQL WHERE clauses, Excel filters, CTRL+F with "whole word only." That's all Boolean logic.

**Here's how it worked in 1970s libraries and early search engines.**

You type a query like: `maternity AND leave AND policy NOT termination`

The system scans every document. If a document contains the word "maternity" AND the word "leave" AND the word "policy" AND does NOT contain "termination" — it's returned. Otherwise: nothing. No gradations, no scores, no ranking. Binary. In or out.

*[Point to the left card]*

This was **revolutionary at the time**. Before this, finding a document in a library catalog required knowing the exact title or author. Boolean search let you search by topic for the first time. MEDLINE — the medical literature database — launched in 1971 with Boolean search. It made the entire medical literature searchable.

*[Point to the right card]*

**Why it failed for general use:**

**No ranking.** You get all matching documents with equal weight. Search for "leave policy" in a database of 100,000 HR documents and you might get 8,000 results. All equal. Which do you read first? No idea.

**No synonyms.** Your query says "car." The perfect document says "automobile." Zero match. Boolean search doesn't know these words are related. It only sees characters.

**Brittle.** One typo in your query — one spelling mistake — zero results. The system has no concept of "close enough." This required trained librarians to construct queries correctly. It was not for regular users.

**No context.** `python AND programming` matches a document that says "python is a snake" in one paragraph and "programming is hard" in another paragraph — even if the document has nothing to do with the Python language.

🗣️ **Ask the room:** "How many of you have ever done CTRL+F in a PDF to find something?" *[Everyone]* "That's Boolean search. Exact character match. You've all experienced its strength — and its limitations — when the word you're looking for is phrased differently in the document."

**The transition to TF-IDF solved the ranking problem. Let's see how.**

---

## 🎬 SLIDE 13 — TF-IDF · *0:41 · 4 min*

Let's understand TF-IDF properly. This is the foundation that BM25 and even modern hybrid systems are built on.

The formula: TF-IDF of a term in a document equals TF times IDF.

*[Point to the TF card]*

**TF — Term Frequency.** How often does a word appear in this specific document, divided by the total words. If "policy" appears 10 times in a 200-word document, TF is 0.05. More occurrences means more relevance.

*[Point to the IDF card]*

**IDF — Inverse Document Frequency.** How rare is this word across ALL documents? You take the log of total documents divided by how many documents contain this word.

Here's the intuition: the word "the" appears in every document — so log(N/N) = log(1) = 0. Zero contribution. The word "maternity" might appear in only 3 documents out of 10,000 — so log(10000/3) is a big number. Rare words are more informative. They tell you more about what a document is about.

💡 Think of it like this: if someone walks into a library and asks about "maternity leave," the word "the" tells you nothing about which shelf to look on. The word "maternity" tells you a lot. TF-IDF formalizes this intuition mathematically.

**The critical limitation:** if you search for "parental leave" and the document says "maternity policy," TF-IDF gives you a score of zero. Zero shared words. The words "parental" and "maternity" mean related things, but TF-IDF doesn't know that. It only sees characters.

📄 **Karen Spärck Jones, 1972, Journal of Documentation.** A computer scientist who literally invented this idea 50 years ago. She said: "some words should count less because they appear everywhere." Published in 1972 and still being used today. That's how fundamental this insight was.

⚠️ Modern note: the basic IDF formula can divide by zero if a word appears in every document. Modern implementations use smoothed IDF: log(N divided by df+1) plus 1. That's what scikit-learn uses.

---

## 🎬 SLIDE 14a — BM25: The Formula · *0:42 · 2 min*

BM25 is TF-IDF with two important improvements. It's what Elasticsearch uses by default. If you've ever used a search engine built on Elasticsearch, you've been using BM25.

*[Point to the formula]*

Don't be scared by this formula. Focus on what's different from TF-IDF. There are two new elements.

*[Point to the left card]*

**First improvement: term saturation — controlled by k₁.**

In TF-IDF, if a word appears 100 times it gets 100 times the score of a word that appears once. That's not realistic. The difference between "policy" appearing once versus twice is huge. The difference between 99 times and 100 times? Almost nothing.

BM25 adds a ceiling. After a certain point, more occurrences stop adding to the score. The k₁ parameter controls where that ceiling kicks in. Default is around 1.2 to 2.0.

*[Point to the right card]*

**Second improvement: length normalization — controlled by b.**

Imagine "policy" appears 3 times in a 2-page memo and 50 times in a 100-page manual. Which is more *about* policy? The memo. BM25 normalizes by document length. Default b is 0.75.

💡 In Italian legal documents, the same term appears many times just because of sentence structure. Without length normalization, long documents would get unfairly high scores. Length normalization fixes this.

📄 **Robertson and Zaragoza, 2009, Foundations and Trends in Information Retrieval.**

---

## 🎬 SLIDE 14b — BM25 Scoring in Action · *0:44 · 2 min*

*[Point to the BM25 Scoring in Action table]*

Now let me show you exactly how BM25 produces a number. This connects the formula to real retrieval.

**The setup:** query is "sick leave policy." The inverted index found 6 candidate chunks. Now BM25 scores each one.

**Step 1: compute IDF for each query term** (how rare is this word across all 100,000 chunks?):
- "sick" — appears in only 200 chunks → **IDF = 4.2** (rare, valuable signal)
- "leave" — appears in 4,000 chunks → **IDF = 2.1** (moderately common)
- "policy" — appears in 20,000 chunks → **IDF = 1.8** (very common, less signal)

**Step 2: for each chunk, multiply IDF × normalized TF, sum across query terms.**

chunk3 (all 3 terms): 4.2×0.8 + 2.1×0.8 + 1.8×0.8 = **6.48** 🏆
chunk41 (only "sick"): 4.2×0.8 = **3.36** — "sick" is rare, so even one term scores well
chunk1 (only "policy"): 1.8×0.8 = **1.44** — common word, lowest score

**Three rules:**
1. **More query terms matched → higher score.** chunk3 (3 terms) beats chunk41 (1 term).
2. **Rarer words give a bigger bonus.** "sick" (IDF 4.2) > "policy" (IDF 1.8).
3. **More occurrences in a short chunk → higher.** But with a cap — the 50th occurrence adds almost nothing.

🗣️ **Ask the room:** "If I add a chunk with 'sick' repeated 500 times — does BM25 give it a huge score?" *[Wait]* "No — saturation. That's k₁ at work."

**Still the same limitation:** "vacation days" and "time off" don't match. BM25 is smarter than TF-IDF but still blind to meaning. That's why we need dense retrieval.

---

## 🎬 SLIDE 15 — From Formula to Results: How Retrieval Works · *0:46 · 3 min*

Before I explain the mechanics, I need to clarify one important terminology point.

**In classical information retrieval textbooks, the word "document" means one article, one webpage, one file.** But in RAG — in *our* system — a "document" in the retrieval sense is always a **chunk**. When we split your 50-page HR manual into 200 chunks of 400 tokens each, each of those 200 chunks becomes its own retrieval unit. It gets its own entry in the inverted index. It gets its own embedding vector. It gets its own row in the vector database.

So when I say "BM25 finds the most relevant documents" — I mean "BM25 finds the most relevant *chunks*." TF-IDF, BM25, cosine similarity — they all operate on **chunks**.

**This is exactly why chunking quality matters so much.** Bad chunks → bad retrieval → bad answers — regardless of which retrieval method you use.

Now — how does retrieval actually find the right chunks?

*[Point to the left card — BM25]*

**BM25 / TF-IDF uses an inverted index.**

Think of an inverted index like the index at the back of a textbook. Instead of "page 47 mentions Python, TF-IDF, and BM25", it's flipped: "Python → appears on pages 3, 47, 112. TF-IDF → appears on pages 47, 88, 201."

In a RAG system, every term gets a posting list: a list of all **chunk IDs** that contain that term. The inverted index is built once, offline, when you load and chunk your documents.

At query time:
1. User asks: *"sick leave policy"*
2. System looks up each query word:
   - "sick" → chunk3, chunk7, chunk41
   - "leave" → chunk3, chunk7, chunk12, chunk88
   - "policy" → chunk1, chunk3, chunk7
3. Candidate set: 6 chunks out of 100,000
4. Compute BM25 score for each candidate **chunk**
5. Rank → return top-k chunks to the LLM

🗣️ **Key insight:** BM25 never even looks at chunks that share zero words with the query. The inverted index eliminates 99%+ before any scoring happens.

*[Point to the right card — Dense]*

**Dense retrieval uses a vector index** — we'll see exactly how this index works in the next few slides.

1. User asks: *"sick leave policy"*
2. Embed the query → a 384-dimensional vector
3. Search the vector index for the nearest **chunk** vectors geometrically — no word lookup at all
4. Return the top-k **chunks** by cosine similarity to the LLM

Notice: no inverted index, no word matching. The system compares the *direction* of the query vector against all stored chunk vectors and returns the closest ones. How it does this fast — without comparing to every single vector — is what we'll cover next.

**The fundamental difference:** BM25 finds chunks by matching *words*. Dense retrieval finds chunks by matching *geometry* — direction in vector space.

**Takeaway: chunk = unit of retrieval in RAG. Both methods index chunks, score chunks, and return chunks — never the original full file.**

---

## 🎬 SLIDE 16 — Dense Retrieval · *0:49 · 4 min*

This is the big leap. Everything we've covered until now — TF-IDF, BM25 — was about matching words. Dense retrieval is fundamentally different: we match **meaning**.

Let me make sure this is crystal clear, because it's the heart of why RAG works.

*[Point to the sparse card]*

**In BM25, a document is a bag of words.** Imagine Excel has a column for every word in the Italian language — that's roughly 50,000 columns. For each document, you fill in how many times each word appears. Most cells are zero because most words don't appear. That's what we call *sparse* — mostly zeros. If the query word doesn't appear in the document, the column is zero. End of story.

*[Point to the dense card]*

**In dense retrieval, a document is a meaning direction.** We don't count words. We run the entire sentence through a neural network — a BERT-style encoder — and get a vector of 384 numbers. Every one of those 384 numbers is non-zero. They represent the *semantic content* — the meaning — of the sentence. That's why it's called *dense*.

Now here is the key insight. **Text with similar meaning produces vectors that point in a similar direction in this 384-dimensional space.** Two sentences that mean the same thing, even if they share zero words, will have vectors that point roughly the same way.

We measure this with **cosine similarity**. Cosine is the angle between two vectors.
- If two vectors point in exactly the same direction: angle = 0°, cosine = 1. Identical meaning.
- If they point at a right angle: cosine = 0. Completely unrelated.
- If they point in opposite directions: cosine = -1. Opposite meaning.

*[Point to the example on the slide]*

"Vacation days" → encoder → vector [0.21, -0.45, 0.78, ...]
"Time off" → encoder → vector [0.19, -0.41, 0.82, ...]

Those two vectors point in almost the same direction. Cosine similarity = 0.89. BM25 score between them = 0 (no shared words). Dense retrieval finds the connection. BM25 is blind to it.

💡 **Another way to think about it:** imagine plotting all words and phrases on a big map. Words with similar meanings live in the same neighborhood. "Time off," "vacation," "days off," "annual leave" are all in the same neighborhood. When you embed a query, you drop a pin on the map and look for documents in the same neighborhood. That's cosine similarity.

💡 **Technical note:** sentence transformers output unit-normalized vectors — each vector has length exactly 1. When all vectors have length 1, the cosine formula simplifies to just the dot product: A·B. No division needed. That's why FAISS and ChromaDB use dot products internally — it's the same math, but faster.

---

## 🎬 SLIDE 17 — The Scale Problem: Why We Need ANN · *0:53 · 1 min*

Quick important question: **"You have 1 million chunk vectors. Do you compute cosine similarity with all of them every time someone asks a question?"**

If you did — brute force, compare against every single vector — it takes about 300 milliseconds. That's too slow for any real application.

The solution is called **ANN — Approximate Nearest Neighbor search.** Instead of checking every vector, you build a smart index that creates shortcuts through the vector space. Result: about 20 comparisons instead of 1 million. From 300ms down to 1ms.

The most popular ANN algorithm is called **HNSW — Hierarchical Navigable Small World.** It's what ChromaDB, Qdrant, and virtually every vector database uses. We'll explain exactly how it works in **Section 4** when we cover vector databases in detail. For now, just know: **dense retrieval is fast because of HNSW, not because cosine similarity is cheap.**

---


## 🎬 SLIDE 18 — Two Retrieval Approaches · *0:52 · 3 min*

This slide is about a choice you will make when building a RAG system. Two approaches. Different trade-offs. Let me explain both clearly.

*[Point to the left card — Bi-Encoder]*

**Bi-encoder — this is the default. Start here.**

The query and the chunk are embedded **separately** using your embedding model — any model: MiniLM, BGE-M3, OpenAI, whatever you choose. You get one vector for the query, one vector for the chunk, and you compute cosine similarity between them.

**Why is this fast?** Because all chunk embeddings are precomputed offline when you build the index. At query time you do exactly one embedding call — for the query only — then an ANN search in milliseconds. That's it. The 100,000 chunk embeddings were already computed hours ago.

**The limitation:** the query and chunk never actually "interact." The model sees them in complete isolation. Relevance = how close their vectors are geometrically. This works very well in practice, but it can miss subtle matches.

*[Point to the right card — Cross-Encoder]*

**Cross-encoder — more accurate, but costly.**

Here, the query and chunk are fed into the model **together** as a single combined input. The model processes them jointly and outputs one relevance score — a number like 0.94 meaning "very relevant" or 0.12 meaning "not relevant."

**Why is it more accurate?** Because the model can compare query words directly against chunk words during processing — it sees the full context of both at once. It truly "reads" the pair.

**Why is it expensive?** Because you cannot precompute anything. You don't know the query until a user asks it. So for every query, you must run the model once per candidate chunk. At query time.

*[Point to the production flow diagram]*

**The solution: combine both.**

You don't have to choose one or the other. In production:
1. Bi-encoder does the heavy lifting — filters 100,000 chunks down to 100 candidates in milliseconds
2. Cross-encoder re-ranks only those 100 — runs 100 times, not 100,000 times

The bi-encoder is fast but approximate. The cross-encoder is slow but precise. Together: fast AND precise.

🗣️ **My recommendation to this room:** **start with the bi-encoder alone.** It's simpler to build, faster, and in most cases good enough. Only add a cross-encoder re-ranker if you measure that retrieval quality is hurting your answers. Don't add complexity before you need it.

---


## 🎬 SLIDE 19 — Embedding Models · *0:55 · 2 min*

This slide shows where the embedding model landscape actually stands today — because it moves fast.

*[Point to the table]*

**Five models. Let me walk through them.**

**MiniLM-L6-v2** — this is what we use in the lab. It's fast, runs on any laptop, zero setup. But be honest with your students: as of 2025, MiniLM is considered legacy. We use it in the lab because it's simple. For any real project, you'd use something better.

**Qwen3-Embedding-0.6B** — this is the modern replacement for MiniLM. Same idea — small, free, runs locally — but dramatically better quality. 1024 dimensions instead of 384, and supports up to 32,000 tokens per chunk. If you're starting a new project today and want something lightweight and free, use this instead of MiniLM.

**BGE-M3** — this is the production open-source workhorse. What makes it unique: it's the only major model that does dense retrieval, sparse retrieval, AND hybrid retrieval — all in one model. You don't need to run a separate BM25 pipeline alongside it. It handles 100+ languages natively, including Italian. Free, runs locally, data stays with you.

**text-embedding-3-large** — OpenAI's cloud model. The safe choice when you want the best quality without managing infrastructure. One interesting feature: it supports **Matryoshka Representation Learning** — you can truncate the 3072 dimensions down to, say, 512, and lose very little quality but save a lot of storage and speed. Paid per API call.

**Gemini Embedding 2** — currently one of the top performers on benchmarks. Supports 32,000 token context, works on text, images, and video. Google's cloud offering.

*[Point to the two cards]*

**The practical decision is simple:** self-hosted or cloud? If your data is sensitive or you need cost predictability at scale → BGE-M3. If you want easy, managed, high quality → text-embedding-3-large or Gemini Embedding 2.

*[Point to the takeaway]*

**Two critical points to remember:**

First — index model must equal query model. Non-negotiable. If you switch models, you must re-index everything.

Second — **don't blindly trust MTEB leaderboard rankings.** MTEB is a great shortlist tool, but the top model on MTEB may not be the best for your specific domain. Legal documents, medical reports, Italian HR policies — these need to be tested on your own data with a tool like Ragas.

**For this course: MiniLM in the lab, BGE-M3 for Italian production. The pipeline code changes by exactly one line.**

---



## 🎬 SLIDE 20 — Hybrid Retrieval · *0:57 · 2 min*

In practice, production systems almost always combine BM25 and dense retrieval. Neither alone is enough.

**When BM25 wins:** exact terms matter. Someone searches "Article 47 of the employment contract" — they want the text that contains exactly those words. Dense retrieval might return "Article 12" because it's semantically related to employment contracts. Wrong. BM25 is exact.

**When dense retrieval wins:** the user expresses something differently than the document does. "What happens if I'm late?" should match "penalties for tardiness." Zero shared words. BM25 scores zero. Dense retrieval scores high.

**Hybrid combines both** with a tuning parameter alpha:

`score(q,d) = α · BM25(q,d) + (1-α) · cosine(embed(q), embed(d))`

α = 0 means trust only dense. α = 1 means trust only BM25. α = 0.4 means 40% BM25, 60% dense.

*[Walk through the example on the slide]*

Query: "sick leave regulations." BM25 score (exact word match): 0.72. Dense score (semantic match to illness/absence/medical leave): 0.85. Hybrid with α = 0.4: 0.4 × 0.72 + 0.6 × 0.85 = 0.799. You get the benefit of both.

⚠️ **The normalization trap — this is a common mistake in real projects.** BM25 scores have no fixed range — they can be 0, 5, 12, 30, or higher depending on document length and term frequency. Cosine scores are always between 0 and 1. If you add them directly: BM25 = 15, cosine = 0.9, sum = 15.9. The BM25 completely dominates. You might as well not have the dense score at all. You must normalize both to the same [0,1] range first using min-max scaling, then combine. Lecture 5 covers Reciprocal Rank Fusion — a technique that avoids this normalization problem entirely.

📄 **Craswell et al., SIGIR 2021 — MS MARCO.** The largest public benchmark for IR shows hybrid consistently beats either method alone.

---

---

# ═══ SECTION 3: DOCUMENT PROCESSING & CHUNKING ═══
## *15 minutes total · Clock: 0:54 → 1:09*

---

## 🎬 SLIDE 21 — Section 3 Intro · *0:54 · 1 min*

Section three. Document processing and chunking.

This is the section that surprises most people. They build a RAG system, the retrieval doesn't work well, they spend days tuning the embedding model. Then they go back and fix their chunking strategy and suddenly everything works.

**Chunking is often the highest-leverage thing you can change in a RAG system. Most people treat it as an afterthought.**

---

## 🎬 SLIDE 22 — Why Chunking Matters · *0:55 · 3 min*

Let me explain why we need to chunk at all.

*[Point to the top card]*

A 50-page HR manual is about 25,000 tokens. MiniLM-L6-v2, our lab model, has a maximum input of **256 tokens** — not 512. We literally cannot embed the whole document as one unit. *(Note: the underlying BERT architecture supports 512 positions, but MiniLM-L6-v2 was trained on sequences of 256. Going beyond that degrades quality.)*

But even if your embedding model supports long inputs, there's still a problem.

*[Point to the two cards at the bottom]*

**The dilution problem — the paint analogy.** Think about mixing paint. If you mix blue and yellow, you get green. If you mix 50 different colors together, you get grey. Grey doesn't match anything specifically. The same thing happens with embeddings. If you embed a 50-page manual covering vacation days, sick leave, maternity policy, remote work, expense reports and 40 other topics — the embedding becomes an average of everything. It matches nothing well. When someone asks about sick leave, this grey averaged vector won't score high for that specific topic.

**The granularity problem.** The user asks a specific question about sick leave. They need those three paragraphs on page 12. You don't want to retrieve the entire 50-page manual — that fills the entire LLM context window with mostly irrelevant content.

**Good chunking = good retrieval = good answers. Bad chunking = bad answers, no matter how good your LLM is.**

---

## 🎬 SLIDE 23 — Three Chunking Strategies · *0:58 · 4 min*

Three main approaches.

*[Point to card 1]*

**Strategy one: fixed-size chunking.** Split every N characters or tokens. Simple to implement. Terrible in practice. You cut mid-sentence, mid-word. You get chunks like "Employees are entitled to 5 months of mand" — the word "mandatory" gets split in half. Meaningless. Don't do this.

*[Point to card 2 — the highlighted one]*

**Strategy two: recursive structural chunking.** The recommended approach. Idea: try to split on the most natural boundary. First try to split at paragraph breaks — double newlines. If the paragraph is still too long, split at sentence boundaries — periods. If still too long, split at commas. You go from big to small, always trying to preserve meaning.

This is what LangChain's RecursiveCharacterTextSplitter does. You define a list of separators in order of preference. The result is chunks that are complete thoughts — they make sense on their own.

*[Point to card 3]*

**Strategy three: semantic chunking.** Instead of splitting at character boundaries, you use embeddings to detect topic shifts. Embed each sentence. Compute similarity between adjacent sentences. When the similarity drops below a threshold θ — the sentence is about a different topic — start a new chunk.

The formula: split at position i if cosine(embedding of sentence i, embedding of sentence i+1) is less than threshold theta.

Most accurate but most expensive — you need to embed every sentence just to decide where to split. And you need to tune theta carefully.

**Default recommendation: recursive structural chunking. Best balance of quality and simplicity.**

---

## 🎬 SLIDE 24 — Chunk Size Trade-Off · *1:02 · 3 min*

Even with the right strategy, you need to pick the right size.

*[Point to the left red card]*

**Too small — 50 to 100 tokens.** High precision but no context. You retrieve "must submit the request to" — submit what? To whom? The LLM can't give a good answer from an incomplete fragment.

*[Point to the right red card]*

**Too large — 1000+ tokens.** Full context but diluted embedding. You retrieve a chunk that's relevant to one part of the question but contains lots of irrelevant text. LLM context window gets noisy.

*[Point to the green card]*

**Sweet spot: 300 to 500 tokens, with 50 to 100 token overlap.** The overlap ensures that a sentence at the boundary between two chunks appears in both. Without overlap, a sentence that falls exactly at a split point could be missed by retrieval.

*[Point to the overlap visual on the slide]*

Look at the example: chunk 1 ends with "...mandatory maternity leave. **During this period, they receive 80%**" and chunk 2 starts with "**During this period, they receive 80%** of their salary. Optional parental..." The highlighted part is the overlap zone — it appears in both chunks. If someone searches for "80% salary during leave," both chunks will match because the boundary sentence is in both.

💡 Think of it as a sliding window. Chunk 1 covers tokens 1–400. Chunk 2 covers tokens 350–750, overlapping by 50 tokens. Without that overlap, the sentence at token 400 might get cut in half and lost to both chunks.

---


## 🎬 SLIDE 25 — Chunking in Practice: All Three Approaches · *1:05 · 3 min*

Let me show you exactly what each strategy produces on the same text. All three strategies, same source, same target size of 400 tokens.

*[Read the source text slowly]*

Four sentences: maternity leave entitlement → salary during leave → optional extension → HR submission process. Four facts. Four clear ideas.

*[Point to the left column — Fixed-Size]*

**Strategy 1 — Fixed-size.** I split every 50 characters, no matter what. Look what happens: "Employees are entitled to 5 months of mand" — "atory maternity leave..." — the word "mandatory" is cut in half. Chunk 1 ends with "mand", chunk 2 starts with "atory". If a student searches for "mandatory leave" — neither chunk matches. The word doesn't exist in either one. Completely broken.

*[Point to the middle column — Recursive]*

**Strategy 2 — Recursive.** I try paragraph breaks first — there's one big paragraph, so no split there. Then I try sentence boundaries. Each sentence is well under 400 tokens. So I get: chunk 1 = the entitlement sentence. Chunk 2 = the salary sentence. Chunk 3 = the extension and submission sentences grouped together, still under 400 tokens. Every chunk is a complete idea. Every chunk is independently searchable.

*[Point to the right column — Semantic]*

**Strategy 3 — Semantic.** This is the most sophisticated. I embed each sentence individually and compute cosine similarity between adjacent sentences. Sentence 1 and 2: similarity 0.91 — same topic, keep them together. Sentence 2 and 3: similarity 0.88 — still related. But sentence 3 to sentence 4: similarity drops to 0.41. That's below our threshold of 0.7 — topic shift detected. The first three sentences are about "leave and salary" — they belong together. Sentence 4 is about "the HR submission process" — a different sub-topic. The model found that automatically.

🗣️ **Ask the room:** "Which one would you use?" *[Wait]* "Recursive — always start with recursive. It's simple, fast, and works for 90% of use cases. Only add semantic chunking when you have documents where topics shift within a single paragraph and retrieval is still failing."

**The key insight:** the words in your chunks are exactly what BM25 indexes and what the embedding model encodes. A broken chunk = broken retrieval = wrong answer from the LLM. Chunking is not a detail — it's the foundation.

---


## 🎬 SLIDE 26 — Metadata · *1:07 · 2 min*

One more important idea: metadata.

Each chunk in your vector database is not just text. It's a tuple of three things: the text content, the embedding vector, and metadata.

The metadata can include: source filename, page number, section heading, creation date, department.

Why does this matter? Because you can combine vector search with metadata filtering.

User asks: "What was the remote work policy in 2024?" — Search for semantically similar chunks AND filter to only documents with date = 2024. Much more precise.

But here's the question you should be asking: **how does the system know to filter by 2024?** The user wrote a natural language question. They didn't write SQL. They didn't say `WHERE date = 2024`. So how does "in 2024" in their question become a structured metadata filter?

*[Point to the three boxes on the slide]*

**Three approaches — and this is a real engineering decision you'll make:**

**Approach 1 — LLM parses the query first.** Before retrieval even starts, you send the user's question to an LLM with a prompt like: "Extract structured filters from this query. Return JSON." The LLM reads "What was the remote work policy in 2024?" and returns `{"date": "2024", "topic": "remote work"}`. Then your retriever uses `date = 2024` as a metadata filter and "remote work" as the semantic search query. **This is the most common approach in production today.** It costs one extra LLM call per query, but it's very reliable.

**Approach 2 — UI provides the filter.** The user picks "Year: 2024" and "Department: HR" from dropdown menus in the interface before asking their question. No extraction needed — you have the filters already. Simpler to build, but less flexible for the user.

**Approach 3 — Rule-based extraction.** You write regex patterns that detect dates (like `\b20\d{2}\b` → matches "2024"), department names, or other keywords in the query string. Fast, no LLM cost, but brittle — it misses things like "last year" or "recent policies."

💡 **In our lab, we won't implement metadata filtering — we'll keep it simple with vector search only.** But in a real production system, this is essential. We'll cover the LLM query parsing approach in detail in **Lecture 5** when we talk about query routing and intelligent pre-processing.

User asks: "What are HR regulations about sick leave?" — Search for similar chunks AND filter to department = HR. You avoid retrieving chunks from finance or legal that happen to mention leave for other reasons.

**The key point: metadata filtering doesn't happen by magic. Your system must extract or collect the filters before retrieval runs.**

---

---

# ═══ SECTION 4: VECTOR DATABASES & THE COMPLETE PIPELINE ═══
## *10 minutes total · Clock: 1:09 → 1:19*

---

## 🎬 SLIDE 27 — Section 4 Intro · *1:09 · 1 min*

Section four. Vector databases and the complete pipeline.

Back in Section 2, I showed you that dense retrieval uses cosine similarity to find the closest chunk vectors — and I teased that brute force is too slow. You need an algorithm called ANN to make it fast. I said "we'll cover exactly how in Section 4."

**We're here now.** In this section: first, how HNSW actually works — the algorithm that powers every vector database. Then, which vector databases exist and how to choose. Then, the complete 5-component RAG pipeline — all the pieces together. Finally, how to write the RAG prompt and the common failure modes.

---


## 🎬 SLIDE 28 — The Search Problem at Scale · *1:10 · 2 min*

Let me put concrete numbers on this problem so you feel why it matters.

You have 1 million chunks. Each chunk has a 384-dimensional vector. A user asks a question. You embed the question. Now you need to find which of those 1 million vectors is most similar.

*[Point to the left red card — brute force]*

**Brute-force — K Nearest Neighbors (KNN).** Compare the query vector to every single stored vector. 1,000,000 × 384 = 384 million operations. On a modern CPU: 200–300 milliseconds. For a web app, that's too slow. And this is just 1 million — what about 10 million? 100 million?

*[Point to the right green card — ANN]*

**ANN — Approximate Nearest Neighbor.** Instead of checking every vector, you build a smart index that creates shortcuts. The index knows which vectors are clustered together. When a query arrives, it jumps directly to the relevant region — maybe 20 comparisons instead of a million. From 300ms down to 1ms. 300× faster.

*[Point to the three-row table]*

**There are three families of ANN algorithms.** HNSW — multi-layer graph, coarse to fine. This is what ChromaDB and Qdrant use, and what we'll deep-dive into next. IVF — partitions the space into clusters, only searches nearby clusters. FAISS and pgvector support this. PQ — compresses vectors to save memory, useful at billion scale. FAISS uses this for very large datasets.

**The key trade-off:** all ANN algorithms are *approximate*. They might miss the single mathematically closest vector 2–5% of the time. But they find 95%+ of the right answers. For retrieval, that's perfectly acceptable. You don't need perfection — you need speed.

---

## 🎬 SLIDE 29 — HNSW Deep Dive · *1:12 · 3 min*

Now let's zoom into HNSW — the algorithm most vector databases use. Let me explain the intuition with an analogy.

*[Point to the city analogy card]*

Imagine you're in a city you've never visited and you need to find the best restaurant nearby. You don't walk every single street. You navigate at a high level first.

**Top layer — World map.** Very few nodes with long-range connections. When a query arrives, you start here. Jump immediately to the rough neighborhood — Europe, not Asia. Skip 99% of the world.

**Mid layer — Country map.** More nodes, medium-range connections. Narrow from Europe to the right country, the right city — Milan.

**Bottom layer — Street map.** Every single chunk vector. Short connections between close neighbors. Walk the final few steps to the exact nearest restaurant — return result.

*[Point to the search algorithm card]*

**Now the actual algorithm — greedy navigation.** Step 1: start at a random node on the top layer. Step 2: look at all your neighbors. Move to whichever one is closest to the query vector. Step 3: keep moving until no neighbor is closer than where you are. Step 4: drop to the next layer at the same position. Repeat until you reach the bottom layer. Return the nearest nodes as results.

💡 **Real analogy for this room:** think about how Google Maps finds the fastest route. It doesn't check every possible road in Italy. Highways first, then regional roads, then local streets. Same hierarchical principle.

**Why "approximate"?** Because the greedy algorithm can get stuck in a local optimum — it found the best neighbor in its local area, but there might be an even better match on the other side of the graph that it never visited. This happens 2–5% of the time. For retrieval, that's perfectly acceptable — you don't need the mathematically perfect nearest chunk, you need a good enough one, fast.

**Practical note — the three parameters you can tune:**

You never implement HNSW yourself. When you call `collection.query()` in ChromaDB, it runs HNSW automatically. But understanding these three knobs helps you tune quality vs. speed:

**1. `M` — max connections per node.** Default: 16. Each node in the graph connects to M neighbors. Higher M = more connections = the graph is better connected = fewer dead ends = higher recall. But: more memory per vector and slower index construction. Think of it as: more roads on your city map. More roads = easier to find any destination, but the map file is bigger. For most RAG systems, M=16 is fine. Only increase to 32–64 if you need >99% recall.

**2. `ef_construction` — build-time search width.** Default: 200. When building the graph (at indexing time), how many candidates does the algorithm consider for each node's neighbors? Higher = better quality graph but slower to build. This only affects indexing speed, not query speed. Set it high (200–400) and forget about it — you build the index once.

**3. `ef_search` — query-time search width.** Default: 10–50 depending on the DB. When searching, how many candidates does the algorithm track at each step? Higher = more accurate but slower queries. This is the main quality/speed knob at query time. If your retrieval is missing relevant chunks, try increasing ef_search from 10 to 50 or 100. If your queries are too slow, decrease it.

**Rule of thumb:** start with defaults. If retrieval quality is poor, increase `ef_search` first. If still poor, increase `M` and rebuild the index. Don't overthink it — defaults work for 90% of RAG systems.

📄 **Malkov & Yashunin, IEEE TPAMI, 2020.** This paper is the foundation of virtually every modern vector database — ChromaDB, Qdrant, pgvector, FAISS all use HNSW under the hood.

---



## 🎬 SLIDE 30 — Vector Database Landscape (2025) · *1:15 · 3 min*

The vector database world has changed dramatically in the last two years. The big trend: **every major database now supports vectors**. You no longer need a separate vector database in many cases.

Let me walk you through the three categories.

*[Point to the first card — Purpose-Built]*

**Category 1 — Purpose-built vector databases.** These were designed from the ground up for vector search.

**ChromaDB** — our lab choice. Runs embedded in Python — no server, no setup, just `pip install` and go. Perfect for prototyping and learning. But: single-process, no real persistence guarantees, not for production.

**Qdrant** — the production upgrade. Client-server architecture, advanced metadata filtering (remember our earlier slide?), very fast. Written in Rust. If you graduate from ChromaDB, this is the first stop.

**Pinecone** — fully managed cloud. You don't run anything. Just API calls. Enterprise pricing. Best for teams that don't want any infrastructure at all.

**Weaviate** — interesting for multi-modal use cases. It has hybrid search (keyword + vector) built in, and can handle images and text together. Also supports GraphQL queries.

*[Point to the second card — Traditional DBs]*

**Category 2 — Traditional databases with vector extensions.** This is the 2025 trend and it's huge.

**PostgreSQL + pgvector** — if your company already runs Postgres (and most do), you just add the pgvector extension. No new infrastructure. Your vectors live next to your relational data. Supports both IVFFlat and HNSW indexes.

**MongoDB + Atlas Vector Search** — same idea for MongoDB shops. If your application data is in MongoDB, you can add vector search without a separate database. The vectors are stored alongside your documents. HNSW-based.

**Elasticsearch + kNN search** — from version 8.x onward, Elasticsearch supports dense vector search natively. If you're already using Elastic for text search, you can add semantic search without any new infrastructure. This gives you hybrid search (BM25 + vectors) for free.

**Neo4j + Vector Index** — this is for **GraphRAG** scenarios. Neo4j is a graph database — it stores relationships between entities. With the vector index feature, you can combine graph traversal with vector similarity. Example: "Find documents similar to this query that were written by people in the same department as the user." The graph finds the relationship, the vector finds the content.

🗣️ *[If students ask "what is GraphRAG?"]*: It's a pattern where you combine knowledge graphs (who is connected to whom, which document references which) with vector retrieval. Instead of just searching by similarity, you also navigate relationships. Very powerful for enterprise use cases. We'll mention it briefly in Lecture 5, but it's a whole advanced topic on its own.

*[Point to the third card — Libraries]*

**Category 3 — Libraries.** These are not databases — they don't persist data or manage a server. They're algorithms you embed in your own code.

**FAISS** — from Meta. The fastest ANN library. Supports GPU acceleration. Used when you have billions of vectors and need raw speed. The research standard.

**Annoy** — from Spotify. Uses random projection trees instead of HNSW. Read-only after building the index. Good for static datasets that don't change.

**The decision guide — and this is important for your projects:**

New project, no existing infrastructure → start with **ChromaDB** for prototyping, graduate to **Qdrant** for production.

Already have **Postgres** → use **pgvector**. Don't add new infrastructure.

Already have **MongoDB** → use **Atlas Vector Search**. Same reason.

Already have **Elasticsearch** → add kNN search. You get hybrid search for free.

Need **graph relationships** → **Neo4j** with vector index.

Need **billion-scale** → **FAISS** (but you'll need to build your own persistence layer).

---


## 🎬 SLIDE 31 — The Complete Pipeline · *1:17 · 2 min*

Let's put all five components together. This is exactly what you'll build in the lab.

**1. Document Loader** — reads your source files. We use `pypdf` for PDFs. In production you'd handle Word docs, HTML, emails, database records.

**2. Text Splitter** — recursive chunking, 400 tokens per chunk, 80 token overlap. This is the design decision with the highest impact on system quality.

**3. Embedding Model** — MiniLM-L6-v2, 384 dimensions. Every chunk gets converted to a 384-number vector.

**4. Vector Database** — ChromaDB in the lab. Stores the vectors and the original text. When a query comes in, it runs HNSW search and returns the top-k results.

**5. Generator (LLM)** — Gemini 2.0 Flash. Takes the retrieved chunks plus the question and generates a grounded answer.

⚠️ **The most common production mistake — the embedding mismatch.** I want to explain this very clearly because it causes real bugs.

Every embedding model creates its own private coordinate system. Think of it like this: Italian and French are both languages, but a word in Italian doesn't mean the same thing in a French dictionary. Similarly, the word "policy" embedded by MiniLM and embedded by text-embedding-3-large produce completely different numbers. They live in different spaces. They are incompatible.

So if you index all your documents with MiniLM, and then someone updates your code and accidentally calls text-embedding-3-large to embed the query, the similarity scores will be garbage. Random numbers. Your retrieval will fail silently — it will return chunks, but they'll be the wrong ones.

**Rule: the embedding model used at index time must be the exact same model used at query time. Version included. Always.**

---

## 🎬 SLIDES 32 & 33 — Prompt Design + Failure Modes · *1:18 · 2 min*

Two quick but important slides.

*[Slide 29 — Prompt Design]*

You already learned prompt engineering in Lecture 3. Now apply it to RAG with three specific rules.

**Grounding:** "Answer ONLY based on the context provided." Without this, the model ignores your retrieved chunks and answers from training knowledge. Back to hallucination.

**Refusal:** "If the answer is not in the context, say: I don't have enough information." *(Note: the slide says "I don't have enough information" — not "I don't know" — use the exact wording).*

**Citation:** "Always cite which document the information comes from."

*[Point to the code block on the slide:]*

The slide also shows the actual Python f-string template: `prompt = f"""Answer ONLY from the context below. If not found, say: I don't have this information. Cite the source. Context: {context}. Question: {question}"""`. This is exactly what students will use in the lab.

*[Slide 30 — Failure Modes]*

When RAG doesn't work, here are the usual causes: wrong chunks — embedding model not good enough or need hybrid search. Chunks too small — add context. Chunks too large — reduce size. No relevant docs — add proper refusal. Model ignores context — strengthen grounding.

And the last one — **stale answers.** This happens when your source documents have changed but the vector database still has the old chunks. For example, the company updates the HR policy in January, but your RAG system still retrieves the 2023 version because nobody re-indexed.

**What does re-indexing actually mean in practice?** It means: detect which documents are new or changed → re-chunk those documents → re-embed the new chunks → add or replace the vectors in your database. You don't have to re-embed everything — just the changed ones. There are two common patterns: a scheduled job that runs nightly or weekly ("cron-based"), or an event-driven pipeline that triggers whenever a document is uploaded or modified. For this course, we'll just re-index manually. In production, you automate it.

**Most RAG failures are retrieval failures. Fix retrieval before you touch the LLM.**


---

---

# ═══ SECTION 5: DECISION MATRIX ═══
## *15 minutes total · Clock: 1:19 → 1:34*

---

## 🎬 SLIDE 34 — Section 5 Intro · *1:19 · 1 min*

Section five. The Decision Matrix.

This is the most practical section of the entire lecture. After this, every time someone at work says "let's use AI for this," you'll know exactly which tool to reach for — and you won't waste weeks building the wrong thing.

---

## 🎬 SLIDE 35 — Three Tools, Three Problems · *1:20 · 2 min*

Three tools in your LLM toolkit. Each solves a different problem.

*[Point to each card]*

**Prompting** — you change the instructions. The model doesn't change. No data needed. No compute. Just text. This is always your first attempt.

**RAG** — you give the model access to external knowledge at query time. You need documents to index, and a small amount of compute. Cheap and updates are instant.

**Fine-tuning** — you actually retrain the model on your data. The model's weights change. You need labeled training examples. You need GPUs. Expensive and slow. But you get deep customization of style, format, and domain behavior.

**The escalation principle: always try the simpler approach first. Prompting → RAG → Fine-tuning. Each step multiplies your cost by roughly 10x.**

---

## 🎬 SLIDE 36 — The Decision Matrix Table · *1:22 · 4 min*

*[Go row by row]*

**Setup time.** Prompting: minutes. RAG: a few hours to set up indexing. Fine-tuning: days to weeks.

**Data required.** Prompting: nothing. RAG: your documents. Fine-tuning: labeled input-output pairs — that's hard to create at scale.

**Compute cost.** Prompting: zero. RAG: low. Fine-tuning: high — GPUs needed.

**Update speed.** Prompting: instant, just change the text. RAG: minutes — update a document, re-index. Fine-tuning: hours — need to retrain.

**Hallucination risk.** Prompting: high — can generate anything. RAG: low — answers grounded in real text. Fine-tuning: medium — model can still misremember.

**Best used for.** Prompting: general tasks the model already knows. RAG: factual Q&A, document queries. Fine-tuning: custom style, specialized domain.

**The Decision Matrix slide** has this row that students often miss:

**Customization.** Prompting: Low. RAG: Medium. Fine-tuning: High. This captures that fine-tuning gives you deep customization of model behavior, while RAG and prompting are more constrained.

**Remember: use prompting first. Always. Each escalation multiplies cost by roughly 10x.** *(This exact line is in the slide takeaway.)*

---

## 🎬 SLIDE 37 — The Decision Flowchart · *1:26 · 3 min*

Three questions. Memorize these.

**Question 1: Does the model already know the answer well enough?** If yes — prompting. General tasks like summarization, translation, classification with common categories — the model was trained on this. Just give it good instructions.

**Question 2: Is the knowledge in documents you can provide?** If yes — RAG. Company policies, product manuals, internal knowledge bases, current events — all in documents you can index.

**Question 3: Do you need specific behavior, format, or domain expertise baked in?** If yes — fine-tuning. Brand voice, specific output format, specialized clinical terminology — this is a style and behavior problem. Fine-tuning is the right tool. *[Note: the slide links this to Lecture 9 — "🔧 Fine-Tuning (Lecture 9)"]*

**If still unsure: try RAG plus better prompts first. Fine-tune only if RAG fails.**

💡 In my experience, 80% of enterprise LLM use cases are solved by good prompting plus RAG. Fine-tuning is often reached for too early, before RAG has even been properly tried.

---

## 🎬 SLIDE 38 — Cloud vs Local Inference · *1:29 · 2 min*

One more decision: cloud API vs local inference.

**Setup time.** Cloud: 5 to 10 minutes — get an API key, install a library. Local with Ollama or LM Studio: 15 to 30 minutes on Apple Silicon. Full CUDA GPU driver setup on Linux: 2 to 4 hours.

**Privacy.** Often the deciding factor in enterprise. Cloud: data leaves your premises. Local: data never leaves your machine.

**Quality.** Cloud APIs give you frontier models — GPT-4, Gemini, Claude. Local models are smaller and quantized — good but not as powerful.

**Tools for local:** Ollama, LM Studio, vLLM, and HuggingFace Transformers.

**Start with cloud during development. Move to local when privacy requirements kick in or costs become significant at scale.**

---

## 🎬 SLIDE 39 — Real Scenarios · *1:31 · 3 min*

Let's apply the matrix to real examples.

*[Point to each card]*

**"Summarize this meeting transcript."** → Prompting. The model was trained on summarization. Just give good instructions.

**"Answer questions about our HR policies."** → RAG. The knowledge is in your documents. Classic use case.

**"Customer support chatbot for our product manuals."** → RAG, not prompting! The model doesn't know your specific product. Without RAG it would hallucinate product specifications. This is the most common mistake — people think prompting is enough for product-specific questions. It's not.

**"Auto-categorize incoming support tickets."** → Start with prompting. Give the model the category list and ask it to classify. If accuracy isn't good enough after prompt iteration, then consider fine-tuning on labeled tickets.

**"Generate medical reports in our specific format."** → Fine-tuning. Specific clinical terminology, specific document structure, specific formatting. This is a style and format problem. Fine-tuning is the right tool.

---

---

# ═══ KEY TAKEAWAYS + Q&A ═══
## *5 minutes · Clock: 1:29 → 1:34*

---

## 🎬 SLIDE 40 — Key Takeaways · *1:34 · 3 min*

Seven things to remember.

*[Go through each]*

**One.** RAG = Retrieve, Augment, Generate. Open-book exam for LLMs. The LLM doesn't need to memorize — it reads.

**Two.** Search evolved from keywords to meaning. TF-IDF and BM25 match words. Dense retrieval matches semantic direction in vector space. Hybrid combines both.

**Three.** Chunking is the most impactful thing you can tune. 300-500 tokens, recursive strategy, 50-100 overlap. Bad chunks = bad retrieval = bad answers. No exceptions.

**Four.** Vector databases use HNSW for O(log n) search. The multi-layer map analogy: coarse to fine, continent → country → street. Milliseconds for millions of vectors.

**Five.** Every RAG prompt needs three things: "ONLY use the context" (grounding), "say I don't know if not found" (refusal), "cite your source" (citation). Miss any one of these and your system will hallucinate or mislead.

**Six.** Prompting first — minutes, free. RAG second — hours, cheap. Fine-tuning last — days, expensive. Each step is 10× more expensive than the previous. Choose wisely.

**Seven.** When RAG fails: fix retrieval first. In 80% of cases the problem is chunking or embedding quality, not the LLM. The LLM reads what you give it — if you give it the wrong chunks, no LLM in the world will save you.

---

## 🎬 SLIDE 41 — Questions · *1:37 · 3 min*

That's the theory for today.

*[Pause. Smile.]*

Next lecture we go hands-on. You'll build a complete RAG pipeline from scratch — document loading, chunking, embedding, indexing in ChromaDB, and generation with Gemini. You'll also see advanced techniques: re-ranking with cross-encoders, HyDE, query decomposition. The lab covers both Lecture 4 and Lecture 5 content together.

Any questions before we finish?

🗣️ *[If no questions, prompt them:] "Let me ask you one: given what you learned today, if your company asked you to build a chatbot that answers questions about your product documentation — what would you use and how would you start?" [Wait. Ideal answer: RAG. Build a vector index of the docs. Start with ChromaDB. Use MiniLM or BGE for embeddings.]*

*[Point to the bottom of the Questions slide — it tells students what's next:]*

**Next → Lecture 5: Advanced RAG & Intelligent Routing**
- Theory: Re-ranking · HyDE · Query decomposition · Semantic routing
- 🔬 Combined Lab: RAG Pipeline (L4) + Query Router (L5) — back-to-back hands-on session

Thank you everyone. See you next time.

---

*End of Lecture 4 Teleprompter*
