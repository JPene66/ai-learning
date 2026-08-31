"""
(QUERY lane): take the final, re-ranked chunks
and assemble them into a clearly-labeled context block, then build the
final prompt that will actually be sent to the LLM.

Input:  the re-ranked chunks (from rerank.py) + the user's question
Output: one complete prompt string, ready for llm_call.py

This is its own step because HOW you format the context measurably affects
answer quality - numbered, labeled sources are what make citations (the
next concept) possible at all.
"""

from rerank import rerank
from search import hybrid_search
from storing import get_collection


def assemble_context(chunks):
    """Labels every chunk with a source number, so the LLM (and the final
    answer) can refer back to '[Source 2]' unambiguously."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_prompt(question, chunks, conversation_history=None):
    """
    Builds the full prompt sent to the LLM. If conversation_history is
    provided (a list of (role, message) tuples), it's included so the model
    has multi-turn context too - this is what multiturn.py (step 9) plugs
    into later.
    """
    context = assemble_context(chunks)

    prompt = f"""You are a an HR manager. Using ONLY the evidence below, Use the following context to answer the question.

{context}
"Answer concisely using the context above."

QUESTION: {question}"""

    return prompt


if __name__ == "__main__":
    collection = get_collection()
    question = "What certifications are found in the resume?"

    candidates = hybrid_search(collection, question, n_results=6)
    top_chunks = rerank(question, candidates, keep_top=4)

    prompt = build_prompt(question, top_chunks)
    print(prompt)