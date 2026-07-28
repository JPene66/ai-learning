import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import pypdf
import uuid

st.set_page_config(page_title="PDF RAG Search", layout="wide")
st.title("📄 PDF RAG Search")

# Setup persistent Chroma client
client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # local, no API key needed
)
collection = client.get_or_create_collection(
    name="pdf_docs", embedding_function=embed_fn
)

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

#  Upload + ingest
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing PDF..."):
        reader = pypdf.PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        chunks = chunk_text(full_text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]

        collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")

st.divider()

#  Search
query = st.text_input("Ask a question about the uploaded PDF(s):")

if query:
    results = collection.query(query_texts=[query], n_results=5)

    st.subheader("Top matching chunks")
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        with st.expander(f"{meta['source']} — chunk {meta['chunk']} (score: {1 - dist:.3f})"):
            st.write(doc)

    #  Optional: feed into an LLM for a synthesized answer
    context = "\n\n".join(results["documents"][0])
    st.subheader("Answer (context-only preview)")
    st.info(
    
        "retrieved context above as grounding for the final answer."
    )