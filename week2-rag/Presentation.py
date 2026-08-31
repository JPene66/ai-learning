"""
(QUERY lane): take the LLM's raw answer and the
chunks it was grounded in, and present a clean, final result to the user -
including a visible source list, so every citation in the answer can
actually be traced back to something real.

Input:  the LLM's answer (from llm_call.py) + the chunks used (from rerank.py)
Output: a formatted, user-facing result (printed here; in app.py this
        becomes the Streamlit display instead)
"""

from llm_call import call_llm
from context_assembly import build_prompt
from rerank import rerank
from search import hybrid_search
from storing import get_collection


def present_answer(question, answer, chunks_used):
    """Formats the final output: the answer, followed by a numbered source
    list so a human can verify any [Source N] citation in the text above."""
    print("=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)
    print(answer)
    print("\n--- Sources used ---")
    for i, chunk in enumerate(chunks_used, start=1):
        print(f"[Source {i}] {chunk['source']}")
    print("=" * 60)


if __name__ == "__main__":
    collection = get_collection()
    question = "What certifications are found in the resume"

    candidates = hybrid_search(collection, question, n_results=6)
    top_chunks = rerank(question, candidates, keep_top=4)
    prompt = build_prompt(question, top_chunks)
    answer = call_llm(prompt)

    present_answer(question, answer, top_chunks)