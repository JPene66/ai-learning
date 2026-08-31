"""
document_loader.py
Parses PDFs, web articles, and plain text files into one consistent,
structured format:

    {
        "content": "the raw extracted text",
        "source": "where it came from (filepath or URL)",
        "doc_type": "pdf" | "web" | "text",
        "metadata": {extra details specific to that source type}
    }
"""

import os
import requests
import pdfplumber
from bs4 import BeautifulSoup


def load_pdf(filepath):
    pages_text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)

    full_text = "\n\n".join(pages_text)

    return {
        "content": full_text,
        "source": filepath,
        "doc_type": "pdf",
        "metadata": {
            "page_count": len(pages_text),
            "pages": pages_text,
        },
    }


def load_web_article(url, timeout=10):
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    title = soup.title.get_text().strip() if soup.title else ""
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
    encodings_to_try = ["utf-8", "cp1252", "latin-1"]
    content = None
    used_encoding = None

    for encoding in encodings_to_try:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                content = f.read()
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if content is None:
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
    if source.startswith("http://") or source.startswith("https://"):
        return load_web_article(source)
    elif source.lower().endswith(".pdf"):
        return load_pdf(source)
    elif source.lower().endswith((".txt", ".md")):
        return load_text_file(source)
    else:
        raise ValueError(f"Don't know how to load this source type: {source}")