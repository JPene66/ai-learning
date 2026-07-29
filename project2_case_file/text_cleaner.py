"""
text_cleaner.py
Clean and preprocess raw text - remove repeated headers/footers, normalize
whitespace, and handle encoding artifacts.
"""

import re
import unicodedata
from collections import Counter


def normalize_whitespace(text):
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_encoding_artifacts(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))
    return text


def remove_repeated_lines(pages, min_repeat_fraction=0.5, edge_lines=3, max_boilerplate_length=80):
    """
    Detects repeated header/footer lines across pages. Only considers SHORT
    lines near the top/bottom of each page as candidates, so genuine repeated
    sentences of real content are never mistaken for boilerplate.
    """
    if len(pages) < 3:
        return pages

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
    if pages:
        cleaned_pages = remove_repeated_lines(pages)
        text = "\n\n".join(cleaned_pages)

    text = fix_encoding_artifacts(text)
    text = normalize_whitespace(text)

    return text