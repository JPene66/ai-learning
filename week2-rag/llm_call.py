"""
(QUERY lane): send the assembled prompt to the
LLM and get back the generated answer.

Input:  the full prompt (from context_assembly.py)
Output: the raw text answer from the model

This is deliberately the simplest file in the whole pipeline - by the time
we get here, all the hard work (retrieval, re-ranking, context formatting)
is already done. This file's only job is the actual generation call.
"""

from helpers import ask
from context_assembly import build_prompt
from rerank import rerank
from search import hybrid_search
from storing import get_collection


def call_llm(prompt):
    """A thin wrapper around helpers.ask() - kept as its own function so
    this step is easy to swap out later (e.g. for a different model, or to
    add retry logic) without touching any other file in the pipeline."""
    return ask(prompt)


if __name__ == "__main__":
    collection = get_collection()
    question = "What is the world's population"

    candidates = hybrid_search(collection, question, n_results=6)
    top_chunks = rerank(question, candidates, keep_top=4)
    prompt = build_prompt(question, top_chunks)

    answer = call_llm(prompt)
    print(answer)