"""
(QUERY lane starts here): take the user's
question and embed it, using the EXACT SAME embedding model used during
ingestion (step 2).

This file is intentionally tiny - the concept it teaches is small but
critical: the query has to go through the identical transformation the
documents went through, or the resulting vectors can't be meaningfully
compared to each other in step 5 (search).
"""

from helpers import embed


def get_user_query():
    """In a real app this might come from a Streamlit text box - here it's
    a simple input() so this file can be run and tested standalone."""
    return input("           What is the Definition of a RAG application ").strip()


def embed_query(query_text):
    """
    This calls the SAME embed() function from helpers.py that embedding.py
    used on every document chunk - that's not a coincidence, it's the whole
    point. One embedding model, used consistently on both sides.
    """
    return embed(query_text)


if __name__ == "__main__":
    query = get_user_query()
    query_vector = embed_query(query)

    print(f"\nQuery: {query}")
    print(f"Embedded into a vector of length {len(query_vector)}")
    print(f"First 5 values: {query_vector[:5]}")