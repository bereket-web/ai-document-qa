from __future__ import annotations

from dataclasses import dataclass

from docqa.embeddings import FastEmbedder
from docqa.store import StoredChunk, VectorStore


@dataclass(frozen=True)
class RetrievedSource:
    source_id: str
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    page: int | None
    chunk_index: int
    score: float
    metadata: dict


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


class HybridRetriever:
    """Exact dense cosine retrieval + FTS5 BM25, combined by reciprocal-rank fusion."""

    def __init__(self, store: VectorStore, embedder: FastEmbedder, rrf_constant: int = 60):
        self.store = store
        self.embedder = embedder
        self.rrf_constant = rrf_constant

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20) -> list[RetrievedSource]:
        query = query.strip()
        if not query:
            return []
        chunks = self.store.all_chunks()
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if not by_id:
            return []

        compatible_chunks = [chunk for chunk in chunks if chunk.model_name == self.embedder.model_name]
        dense = []
        if compatible_chunks:
            query_vector = self.embedder.embed_query(query)
            dense = sorted(
                (
                    (cosine_similarity(query_vector, chunk.embedding), chunk.chunk_id)
                    for chunk in compatible_chunks
                    if len(chunk.embedding) == len(query_vector)
                ),
                key=lambda pair: pair[0], reverse=True,
            )[:candidate_k]
        lexical_ids = [chunk_id for chunk_id in self.store.lexical_search(query, candidate_k) if chunk_id in by_id]
        scores: dict[str, float] = {}
        for rank, (_, chunk_id) in enumerate(dense, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_constant + rank)
        for rank, chunk_id in enumerate(lexical_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_constant + rank)
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [self._source(by_id[chunk_id], position, score) for position, (chunk_id, score) in enumerate(ranked, start=1)]

    @staticmethod
    def _source(chunk: StoredChunk, position: int, score: float) -> RetrievedSource:
        return RetrievedSource(
            source_id=f"S{position}", chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, filename=chunk.filename,
            text=chunk.text, page=chunk.page, chunk_index=chunk.chunk_index, score=score, metadata=chunk.metadata,
        )
