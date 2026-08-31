"""
Objective: Parse PDFs, web articles, and plain text files into one
consistent, structured format.

Every loader function below returns the SAME shape of dictionary, no matter
what kind of source it came from:

    {
        "content": "the raw extracted text",
        "source": "where it came from (filepath or URL)",
        "doc_type": "pdf" | "web" | "text",
        "metadata": {extra details specific to that source type}
    }

This consistency is the entire point - objective (cleaning) and objective 6
(the ingestion script) don't need to know or care what kind of file this
started out as, because by the time it reaches them, it's just a dictionary
with the same 4 keys every time.
"""

import os
import requests
import pdfplumber
from bs4 import BeautifulSoup


def load_pdf(filepath):
    """
    Extracts text from a PDF, page by page, using pdfplumber.
    pdfplumber is used over pypdf here because it handles awkward PDF
    layouts (columns, tables) more reliably - worth the extra dependency.
    """
    pages_text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""  # extract_text() can return None on image-only pages
            pages_text.append(text)

    full_text = "\n\n".join(pages_text)

    return {
        "content": full_text,
        "source": filepath,
        "doc_type": "pdf",
        "metadata": {
            "page_count": len(pages_text),
            "pages": pages_text,  # kept separately - useful later for removing repeated headers/footers
        },
    }

def load_web_article(url, timeout=10):
    """
    Fetches a web page and extracts just the readable article text -
    stripping out navigation, scripts, ads, and other page furniture that
    would otherwise pollute your chunks with irrelevant text.
    """
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()  # raises an error immediately on a bad status code (404, 500, etc.)

    soup = BeautifulSoup(response.text, "html.parser")

    # Strip tags that are never part of the actual article content
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    title = soup.title.get_text().strip() if soup.title else ""

    # get_text() with a separator keeps paragraphs from running together into one giant word-blob
    article_text = soup.get_text(separator="\n")

    return {
        "content": article_text,
        "source": url,
        "doc_type": "web",
        "metadata": {
            "title": title,
            "status_code": response.status_code,
        },
    }


def load_text_file(filepath):
    """
    Reads a plain .txt or .md file, handling the encoding issues that
    real-world files (especially ones exported from Word or older systems)
    commonly have.
    """
    encodings_to_try = ["utf-8", "cp1252", "latin-1"]
    content = None
    used_encoding = None

    for encoding in encodings_to_try:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read()
            used_encoding = encoding
            break  # stop at the first encoding that works without erroring
        except UnicodeDecodeError:
            continue  # try the next encoding in the list

    if content is None:
        # Last resort: force-read as UTF-8, replacing any byte that can't be
        # decoded with a placeholder character, instead of crashing entirely.
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        used_encoding = "utf-8 (with replacement characters)"

    return {
        "content": content,
        "source": filepath,
        "doc_type": "text",
        "metadata": {
            "encoding_used": used_encoding,
            "file_size_bytes": os.path.getsize(filepath),
        },
    }

def load_document(source):
    """
    The single entry point for objective 4. Looks at what 'source' is and
    dispatches to the right loader - this is the function the rest of your
    code should actually call.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_web_article(source)
    elif source.lower().endswith(".pdf"):
        return load_pdf(source)
    elif source.lower().endswith((".txt", ".md")):
        return load_text_file(source)
    else:
        raise ValueError(f"Don't know how to load this source type: {source}")


if __name__ == "__main__":
    # Quick manual test, replace with a real file/URL you have on hand
    doc = load_document("C:\\Users\\Jorda\\Downloads\\SJEP RS.UD.pdf")
    print(f"Loaded {doc['doc_type']} from {doc['source']}")
    print(f"Content preview: {doc['content'][:200]}")
    print(f"Metadata: {doc['metadata']}")