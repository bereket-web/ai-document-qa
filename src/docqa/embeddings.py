from __future__ import annotations

from collections.abc import Iterable


class FastEmbedder:
    """CPU-friendly multilingual embeddings backed by FastEmbed/ONNX Runtime."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", threads: int = 2):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies to enable local embeddings.") from exc
        self.model_name = model_name
        self._is_e5 = "e5" in model_name.lower()
        self._model = TextEmbedding(model_name=model_name, threads=threads)

    @staticmethod
    def _normalize(vector: Iterable[float]) -> list[float]:
        values = [float(value) for value in vector]
        norm = sum(value * value for value in values) ** 0.5
        if norm == 0:
            return values
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # E5 instructs callers to distinguish passages from search queries; MiniLM does not use prefixes.
        passages = [f"passage: {text}" if self._is_e5 else text for text in texts]
        vectors = self._model.passage_embed(passages)
        return [self._normalize(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        query = f"query: {text}" if self._is_e5 else text
        vectors = self._model.query_embed([query])
        return self._normalize(next(iter(vectors)))
