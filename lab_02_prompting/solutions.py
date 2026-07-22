"""Lab 02: Solutions — Chain-of-Thought prompt implementation.

This is the reference implementation for the CoT step (Step 5).
Compare it to your own attempt in step_5_cot() inside lab_02_prompting.py.
"""

from textwrap import dedent


def prompt_cot_solution(client, model, review):
    """
    Chain-of-Thought sentiment classification.

    Instead of asking the model to jump straight to the answer,
    we explicitly ask it to reason step by step.
    Each intermediate step narrows the probability distribution
    for the next one — leading to a more accurate final answer.

    Compare with step_3_system_message: that gives you a fast one-word
    answer. CoT gives you an auditable reasoning trail.
    """
    prompt = dedent(f"""\
        Classify this software review as POSITIVE, NEGATIVE, or NEUTRAL.
        Think step by step, analyzing each aspect before your final classification.

        Review: "{review}"

        Step 1 - Identify positive aspects:
        Step 2 - Identify negative aspects:
        Step 3 - Weigh which matters more for a software product:
        Step 4 - Final classification (one word: POSITIVE / NEGATIVE / NEUTRAL):""")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


# ─── Example usage ───────────────────────────────────────────
if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).parent.parent / ".env")
    client = OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model = "gemini-3.1-flash-lite"

    review = "Easy to use but crashes frequently."
    result = prompt_cot_solution(client, model, review)
    print(result)
