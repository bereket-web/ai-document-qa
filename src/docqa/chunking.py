from __future__ import annotations

import re
from dataclasses import dataclass

from docqa.ingest import PageText


@dataclass(frozen=True)
class TextChunk:
    text: str
    index: int
    page: int | None


def chunk_pages(pages: list[PageText], chunk_words: int = 180, overlap_words: int = 35) -> list[TextChunk]:
    if chunk_words < 20:
        raise ValueError("chunk_words must be at least 20")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be non-negative and smaller than chunk_words")

    chunks: list[TextChunk] = []
    step = chunk_words - overlap_words
    for page in pages:
        words = list(re.finditer(r"\S+", page.text, flags=re.UNICODE))
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            excerpt = page.text[words[start].start():words[end - 1].end()].strip()
            if excerpt:
                chunks.append(TextChunk(text=excerpt, index=len(chunks), page=page.page))
            if end == len(words):
                break
            start += step
    return chunks
