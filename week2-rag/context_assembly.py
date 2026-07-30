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

SYSTEM_PROMPT = (
    "You are Case File AI, a careful research analyst. You answer strictly from "
    "the evidence you are given, you always cite it, and you say plainly when "
    "the evidence does not cover something. You never fill gaps from your own "
    "background knowledge."
)


def assemble_context(chunks):
    """Labels every chunk with a source number, so the LLM (and the final
    answer) can refer back to '[Source 2]' unambiguously."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(blocks)


def format_history(conversation_history, max_turns=6):
    """
    Renders prior turns as plain text for the prompt. Only the most recent
    turns are kept - old history costs tokens on every single request and
    rarely changes the answer.
    """
    if not conversation_history:
        return ""
    recent = conversation_history[-max_turns:]
    lines = [
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in recent
        if turn.get("content")
    ]
    return "\n".join(lines)


def build_prompt(question, chunks, conversation_history=None):
    """
    Builds the full prompt sent to the LLM. If conversation_history is
    provided (a list of {"role": ..., "content": ...} dicts), it's included so
    the model has multi-turn context too - this is what multiturn.py plugs
    into.

    The answering rules are explicit and numbered on purpose: "cite your
    sources" as a vague instruction gets ignored far more often than a
    concrete "write [Source 1] directly after the sentence it supports".
    """
    if not chunks:
        # No evidence retrieved - ask for an honest refusal rather than letting
        # the model answer from memory, which is exactly the failure mode RAG
        # is supposed to prevent.
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "No evidence was retrieved from the case file for this question.\n\n"
            f"QUESTION: {question}\n\n"
            "Reply that the case file contains no evidence relevant to this "
            "question, and suggest what kind of document would answer it."
        )

    context = assemble_context(chunks)
    history_block = format_history(conversation_history)

    prompt = f"""{SYSTEM_PROMPT}

EVIDENCE FROM THE CASE FILE:
{context}
"""

    if history_block:
        prompt += f"""
EARLIER IN THIS CONVERSATION:
{history_block}
"""

    prompt += f"""
QUESTION: {question}

ANSWERING RULES:
1. Use ONLY the evidence above. If it does not answer the question, say so directly.
2. Cite every factual claim with the source it came from, written as [Source 1],
   [Source 2] etc., placed immediately after the sentence it supports.
3. If two sources disagree, say so explicitly and cite both sides rather than
   silently picking one.
4. Do not invent source numbers - only cite sources that appear above.
5. Be concise: a short, well-cited answer beats a long, vague one.

ANSWER:"""

    return prompt


if __name__ == "__main__":
    collection = get_collection()
    question = "What certifications are found in the resume?"

    candidates = hybrid_search(collection, question, n_results=6)
    top_chunks = rerank(question, candidates, keep_top=4)

    prompt = build_prompt(question, top_chunks)
    print(prompt)
