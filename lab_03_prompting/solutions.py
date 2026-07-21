"""Lab 03: Solutions — Chain-of-Thought prompt implementation."""


def prompt_cot_solution(client, model, query, context):
    """Chain-of-Thought: force step-by-step reasoning."""
    from textwrap import dedent

    system = dedent("""\
        You are an HR policy assistant. You must answer questions based ONLY on the provided context.
        
        IMPORTANT: Think step by step before answering.
        
        Follow this process:
        1. IDENTIFY: Which sections of the context are relevant to the question?
        2. ANALYZE: What do those sections say? Are there any conditions or exceptions?
        3. REASON: How do the relevant sections combine to answer the question?
        4. ANSWER: Provide a clear, concise final answer.
        5. SOURCES: List the source documents you used.
        
        If the context does not contain enough information, say so clearly.""")

    user = f"""Context:
{context}

Question: {query}

Let's think step by step:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content
