"""
Objective: Clean and preprocess raw text - remove repeated headers/footers,
normalize whitespace, and handle encoding artifacts.

Why this matters: raw extracted text (especially from PDFs) is messy in
predictable ways - page numbers repeated on every page, inconsistent line
breaks, stray unicode characters from copy-paste artifacts. If you chunk and
embed messy text as-is, that noise gets baked into every single chunk.
Cleaning BEFORE chunking (objective) is what makes the rest of the RAG
pipeline actually work well.
"""

import re
import unicodedata
from collections import Counter


def normalize_whitespace(text):
    """
    Collapses messy whitespace into something consistent:
    - Multiple spaces/tabs become a single space
    - 3+ blank lines in a row become just 2 (one clean paragraph break)
    - Leading/trailing whitespace on every line is stripped
    """
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    text = re.sub(r"[ \t]+", " ", text)          # multiple spaces/tabs -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)        # 3+ newlines -> exactly 2 (one paragraph break)

    return text.strip()


def fix_encoding_artifacts(text):
    """
    Normalizes unicode so visually-identical characters are treated as
    identical (e.g. some PDFs use a 'fancy' apostrophe that looks right but
    is a different character underneath), and strips characters that can't
    be reasonably printed or used.
    """
    # NFKC normalization: converts "compatibility" unicode variants into their
    # standard form - e.g. full-width characters, certain accented letters,
    # and typographic quote marks get normalized to consistent equivalents.
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters (invisible artifacts) but keep newlines and tabs
    text = "".join(ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))

    return text


def remove_repeated_lines(pages, min_repeat_fraction=0.5, edge_lines=3, max_boilerplate_length=80):
    """
    Detects and removes lines that repeat across many pages - the classic
    signature of a header or footer (e.g. "Company Confidential - Page 1",
    a document title on every page, etc.).

    Lines are matched by a normalized "signature" (digits replaced with #)
    rather than exact text, so "Page 1", "Page 2", "Page 3" are correctly
    recognized as the same repeating footer, not three unrelated lines.

    Two safety limits keep this from ever deleting real content that simply
    happens to repeat (e.g. a templated report where every page legitimately
    starts with a similar sentence):
    - Only lines within the first/last `edge_lines` of a page are even
      considered candidates - real headers/footers live at the page edges,
      not buried in the middle of a paragraph.
    - Only SHORT lines (under `max_boilerplate_length` characters) can be
      classified as boilerplate - headers/footers are brief by nature; a
      full repeated sentence is far more likely to be genuine (if unusual)
      content than a page header.

    pages: a list of page-text strings, e.g. doc["metadata"]["pages"] from
    document_loader.load_pdf()
    """
    if len(pages) < 3:
        return pages  # not enough pages to reliably detect a repeated pattern

    def signature(line):
        return re.sub(r"\d+", "#", line.strip())

    signature_counts = Counter()
    for page in pages:
        lines = [l for l in page.split("\n") if l.strip()]
        edge_candidates = lines[:edge_lines] + lines[-edge_lines:]
        unique_signatures_on_page = {
            signature(line) for line in edge_candidates
            if len(line.strip()) <= max_boilerplate_length
        }
        signature_counts.update(unique_signatures_on_page)

    threshold = len(pages) * min_repeat_fraction
    boilerplate_signatures = {sig for sig, count in signature_counts.items() if count >= threshold}

    cleaned_pages = []
    for page in pages:
        kept_lines = [
            line for line in page.split("\n")
            if signature(line) not in boilerplate_signatures
        ]
        cleaned_pages.append("\n".join(kept_lines))

    return cleaned_pages


def clean_text(text, pages=None):
    """
    The main entry point for objective 5. Runs the full cleaning pipeline:
    1. Remove repeated headers/footers (if page-level data is available)
    2. Fix encoding artifacts
    3. Normalize whitespace

    If 'pages' is provided (from a PDF), header/footer removal runs first
    and the cleaned pages are rejoined before the rest of the cleaning steps.
    """
    if pages:
        cleaned_pages = remove_repeated_lines(pages)
        text = "\n\n".join(cleaned_pages)

    text = fix_encoding_artifacts(text)
    text = normalize_whitespace(text)

    return text


if __name__ == "__main__":
    messy_pages = [
        "ACME CORP CONFIDENTIAL\nIntroduction\nThis report covers Q1 results.\nPage 1",
        "ACME CORP CONFIDENTIAL\nRevenue grew by 12% year over year.\nPage 2",
        "ACME CORP CONFIDENTIAL\nWe expect continued growth in Q2.\nPage 3",
    ]
    cleaned = clean_text("", pages=messy_pages)
    print(cleaned)
    # "ACME CORP CONFIDENTIAL" and "Page N" lines should be gone - they
    # appeared on every page and are correctly identified as boilerplate.