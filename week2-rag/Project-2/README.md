# 🗂️ Case File AI

A retrieval-augmented research assistant over a **case file** — a mixed collection of PDFs, web articles and text files.

It doesn't just find passages. It **answers** the question with citations, tells you **when your sources disagree with each other**, and gives you an **explainable confidence score** for the result.

---

## What it does

| Stage | What happens | Where it lives |
|---|---|---|
| Load | PDF, `.txt`/`.md`, or web URL → one common format | `document_loader.py` |
| Clean | Strips repeated headers/footers, normalizes whitespace + unicode | `text_cleaner.py` |
| Chunk | Paragraph-aware chunking with overlap | `ingest.py` |
| Embed | Explicit, batched OpenAI embeddings | `helpers.py` |
| Store | Persistent Chroma collection (cosine distance) | `storing.py` |
| Retrieve | Vector **+** keyword, blended into a hybrid score | `search.py` |
| Re-rank | LLM re-orders the shortlist by true relevance | `rerank.py` |
| Assemble | Numbered `[Source N]` context block + answering rules | `context_assembly.py` |
| Generate | The cited answer | `llm_call.py` |
| Audit | Cross-source contradiction detection | `contradiction.py` |
| Score | Explainable confidence heuristic | `report.py` |
| Follow up | Rewrites follow-up questions before retrieval | `multiturn.py` |
| Grade | Faithfulness / relevance / groundedness | `rag_eval.py` |
| Wire | Calls all of the above in order | `pipeline.py` |

`app.py` is only the Streamlit interface — all pipeline logic lives in the modules above, in the parent folder.

---

## Setup

```bash
cd week2-rag/Project-2

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then put your real key in .env
```

`.env` needs one line:

```
OPENAI_API_KEY=sk-...
```

## Run

```bash
streamlit run app.py
```

Then, in the sidebar: add a PDF / text file / article URL, and ask a question.

## Test

```bash
pytest tests/ -v
```

The tests inject **fake LLMs**, so they run offline, cost nothing, and need no API key.

---

## How to use it

1. **Add evidence** in the sidebar — upload files, or paste an article URL. Each source is listed with its chunk count, and can be removed individually.
2. **Ask a question** in the chat box.
3. **Read the result**:
   - The answer, with `[Source N]` citations you can expand to see the exact passage.
   - ⚠️ **Conflicts** — where two sources say incompatible things.
   - **Confidence**, plus a "Why this score?" breakdown of every factor.
4. **Ask follow-ups** — "and what time was that?" works; the question is rewritten into a standalone one before retrieval.

Contradiction detection is most interesting with **two or more sources covering the same events** — a single document rarely disagrees with itself.

---

## Design decisions worth knowing

**Embeddings are explicit, not automatic.** Chroma can embed documents for you with a built-in local model. This project instead calls `helpers.embed_many()` on the document side and `helpers.embed()` on the query side. That's deliberate: both sides provably use the same model (`text-embedding-3-small`). Letting Chroma do it silently uses a *different* model on each side of the search, which is the kind of bug that produces quietly bad results rather than an error.

**The collection uses cosine distance, explicitly.** Chroma's default is L2, and every similarity score in the UI is computed as `1 - distance` — a formula that's only meaningful for cosine, where distance is bounded in `[0, 2]`. `storing.py` sets `{"hnsw:space": "cosine"}` at creation, and warns if it finds an older collection that isn't using it.

**Hybrid search merges two result sets, it doesn't just re-score one.** A wide vector pool and the top keyword hits are unioned before blending. If keyword scores were only applied to vector results, a chunk containing your exact search term but phrased unusually could never be recovered.

**Confidence is a heuristic, not a model output.** Asking an LLM "how confident are you?" gets you a number it can't justify. This score is computed from things we can observe:

| Factor | Effect |
|---|---|
| Evidence strength (mean of top 3 calibrated retrieval scores) | base score |
| Corroborating sources | +0.05 each, capped at +0.15 |
| Only one source | −0.10 |
| Contradictions | −0.35 high / −0.20 medium / −0.08 low, capped at −0.60 |
| Answer cites nothing | −0.10 |
| Answer honestly says "not in the evidence" | capped at 0.20 |

Every one of those is shown to the user under "Why this score?", and every one is unit-tested in `tests/test_confidence.py`.

**Confidence measures semantic similarity, not the blended rank.** The hybrid `blended_score` deliberately rewards literal word overlap so exact terms aren't missed — that's the right signal for *ranking*. It's the wrong signal for *quality*: a natural-language question shares few literal words with even a perfect answer passage, so blending keyword overlap into the confidence figure systematically deflates it. `report.py` scores on `vector_similarity` instead.

**Retrieval scores are calibrated before use.** With `text-embedding-3-small`, unrelated text still scores ~0.15 cosine similarity and a genuinely on-point passage lands around 0.5–0.6 — it essentially never reaches 1.0. Feeding those numbers straight into the confidence score would report a *perfect* retrieval as "50% confident". `report.py` therefore stretches the observed range (`RELEVANCE_FLOOR` 0.20 → `RELEVANCE_CEILING` 0.60) onto a real 0–1 scale. If you swap embedding models, those two constants are what you retune.

**Failures degrade instead of crashing.** If re-ranking, contradiction detection, evaluation, or follow-up rewriting fails, the pipeline falls back to a sane default and still answers. Only the generation call itself is allowed to surface an error, because there is no answer without it.

**Chunk IDs are deterministic** (`sha1(source)[:12]-index`), and storage uses `upsert`. Re-adding the same document updates it in place instead of storing a second copy; the old version's chunks are cleared first so a shortened document can't leave stale evidence behind.

---

## Notes

- Scanned, image-only PDFs have no text layer. The app says so clearly rather than indexing an empty document — those need OCR first.
- The vector store lives in `Project-2/chroma_db/` and persists across restarts. Delete that folder to start a fresh case file.
- Costs are dominated by ingestion (one batched embedding request per ~64 chunks). Each question costs 2–4 small `gpt-4o-mini` calls: re-rank, generate, contradiction check, and optionally the evaluation.
