from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from docqa.answer import generate_answer
from docqa.chunking import chunk_pages
from docqa.config import Settings
from docqa.embeddings import FastEmbedder
from docqa.ingest import parse_document
from docqa.retrieval import HybridRetriever
from docqa.store import VectorStore


class DocumentQA:
    def __init__(self, settings: Settings | None = None, embedder: FastEmbedder | None = None):
        self.settings = settings or Settings.from_env()
        self.store = VectorStore(self.settings.db_path)
        self._embedder = embedder
        self._retriever: HybridRetriever | None = None

    @property
    def embedder(self) -> FastEmbedder:
        if self._embedder is None:
            self._embedder = FastEmbedder(model_name=self.settings.embedding_model)
        return self._embedder

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None or self._retriever.embedder is not self.embedder:
            self._retriever = HybridRetriever(self.store, self.embedder)
        return self._retriever

    def ingest(self, filename: str, content: bytes) -> dict[str, Any]:
        clean_name = Path(filename.replace("\\", "/")).name
        clean_name = re.sub(r"[^\w.() -]", "_", clean_name, flags=re.UNICODE).strip(" .") or "document.txt"
        digest = hashlib.sha256(content).hexdigest()
        if self.store.has_digest(clean_name, digest):
            return {"filename": clean_name, "status": "already_indexed", "chunks": 0}
        pages = parse_document(clean_name, content, max_bytes=self.settings.max_file_bytes)
        chunks = chunk_pages(pages, self.settings.chunk_words, self.settings.overlap_words)
        embedder = self.embedder
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding model returned an incomplete batch")
        doc_id = hashlib.sha256(f"{clean_name}\0{digest}".encode()).hexdigest()[:24]
        rows = [{
            "chunk_id": hashlib.sha256(f"{doc_id}:{chunk.index}".encode()).hexdigest()[:32],
            "chunk_index": chunk.index,
            "text": chunk.text,
            "page": chunk.page,
            "metadata": {"page": chunk.page, "source_name": clean_name},
        } for chunk in chunks]
        self.store.replace_document(doc_id=doc_id, filename=clean_name, digest=digest, chunks=rows, vectors=vectors, model_name=embedder.model_name)
        return {"doc_id": doc_id, "filename": clean_name, "status": "indexed", "chunks": len(chunks), "sha256": digest}

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Enter a question.")
        hits = self.retriever.retrieve(question, top_k=top_k or self.settings.top_k)
        answer = generate_answer(question, hits, self.settings)
        return {
            "question": question,
            "answer": answer.answer,
            "answer_mode": answer.mode,
            "model": answer.model,
            "sources": [{
                "id": hit.source_id,
                "filename": hit.filename,
                "page": hit.page,
                "chunk": hit.chunk_index,
                "score": round(hit.score, 6),
                "text": hit.text,
            } for hit in hits],
        }

    def status(self) -> dict[str, Any]:
        stats = self.store.stats()
        return {
            **stats,
            "embedding_model": self.settings.embedding_model,
            "embedding_loaded": self._embedder is not None,
            "llm_enabled": bool(self.settings.llm_api_key),
            "llm_model": self.settings.llm_model if self.settings.llm_api_key else None,
            "vector_backend": "SQLite exact cosine + FTS5 BM25 rank fusion",
            "supported_types": ["PDF", "DOCX", "TXT", "MD"],
        }
