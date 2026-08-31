"""
memory.py
Implements all three memory types from the study guide
(Week 3, Skill 1, Concept 3: Agent Memory).

- Short-term: the conversation buffer already living in AgentState
  (conversation_history) - nothing extra needed, included here only as a
  helper for trimming it to the last N messages.
- Long-term: a persistent Chroma collection of past deal notes + outreach
  messages, embedded with Chroma's built-in local model (free, no OpenAI
  embedding cost, no torch/CUDA dependency).
- Episodic: a JSON-backed "lessons learned" log - specific past experiences
  ("last time a fintech lead pushed back on price, X worked") that get
  retrieved and injected into future objection-handling prompts.
"""

import json
import os
import math
import re
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from knowledge_base import PAST_DEALS

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EPISODIC_LOG_PATH = os.getenv("EPISODIC_LOG_PATH", "./episodic_memory.json")


class OfflineHashEmbedding(EmbeddingFunction):
    """
    A fully offline, zero-download, zero-cost embedding function using the
    'hashing trick': each word is hashed into one of N buckets, and the
    resulting bucket-count vector (L2-normalized) stands in for a real
    semantic embedding.

    Why this instead of Chroma's built-in default embedding function or
    sentence-transformers: both require downloading a model file (~90MB+)
    from an external host on first use. That works fine on a normal home
    connection, but fails behind restrictive school/office networks or
    sandboxed environments with no notice beyond a cryptic hash-mismatch
    error - exactly what happened while testing this project. This
    function needs nothing beyond the Python standard library, so it works
    identically everywhere, with zero setup friction.

    Tradeoff to be upfront about: this captures word-overlap similarity,
    not true semantic meaning (it won't know "car" and "automobile" are
    related the way a real embedding model would). For this project's
    retrieval tasks - matching deal notes, case studies, and objections
    that share vocabulary - that's a perfectly reasonable trade for zero
    cost and zero network dependency. Swap in Chroma's DefaultEmbeddingFunction
    or an OpenAI embedding call here if higher-quality semantic matching is
    ever needed and reliable internet access is available.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str):
        vec = [0.0] * self.dim
        words = re.findall(r"[a-z0-9]+", text.lower())
        for word in words:
            bucket = hash(word) % self.dim
            sign = 1.0 if (hash(word) // self.dim) % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_one(text) for text in input]

    @staticmethod
    def name() -> str:
        return "offline_hash_embedding_v1"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict) -> "OfflineHashEmbedding":
        return OfflineHashEmbedding(dim=config.get("dim", 256))


_embed_fn = OfflineHashEmbedding()


def get_long_term_memory(persist_path=CHROMA_PATH):
    client_db = chromadb.PersistentClient(path=persist_path)
    return client_db.get_or_create_collection("apex_long_term_memory", embedding_function=_embed_fn)


def seed_long_term_memory(collection):
    """Loads PAST_DEALS into the vector store, once, so qualification and
    objection-handling have real history to retrieve against."""
    if collection.count() > 0:
        return 0
    collection.add(
        documents=[d["notes"] for d in PAST_DEALS],
        metadatas=[{"industry": d["industry"], "deal_size_usd": d["deal_size_usd"],
                     "outcome": d["outcome"], "objection_handled": d["objection_handled"]} for d in PAST_DEALS],
        ids=[d["id"] for d in PAST_DEALS],
    )
    return len(PAST_DEALS)


def query_long_term_memory(collection, query_text, n_results=3, where=None):
    results = collection.query(query_texts=[query_text], n_results=n_results, where=where)
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return [{"text": d, **m} for d, m in zip(docs, metas)]


def trim_short_term_memory(conversation_history, max_messages=6):
    """Keeps only the last N messages - short-term memory is deliberately
    bounded, unlike long-term memory which persists indefinitely."""
    return conversation_history[-max_messages:]


# --- Episodic memory: a simple append-only JSON log of specific lessons ---

def load_episodic_log(path=EPISODIC_LOG_PATH):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_episodic_lesson(lead_id, objection_type, what_worked, path=EPISODIC_LOG_PATH):
    """Appends one 'lesson learned' entry - e.g. after successfully handling
    a pricing objection, this records WHAT worked so future leads with the
    same objection type benefit from it."""
    log = load_episodic_log(path)
    log.append({"lead_id": lead_id, "objection_type": objection_type, "what_worked": what_worked})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    return log


def recall_lessons_for_objection(objection_type, path=EPISODIC_LOG_PATH):
    """Returns past lessons for this exact objection type, if any exist -
    used to inform the Close agent's next response."""
    log = load_episodic_log(path)
    return [entry for entry in log if entry["objection_type"] == objection_type]