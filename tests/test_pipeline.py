from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from docx import Document

from docqa.chunking import chunk_pages
from docqa.config import Settings
from docqa.ingest import DocumentError, PageText, parse_document
from docqa.retrieval import HybridRetriever
from docqa.service import DocumentQA


class FakeEmbedder:
    model_name = "fake-test-embedding"

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if any(word in lowered for word in ("security", "credential", "password")):
            return [0.0, 1.0, 0.0]
        if any(word in lowered for word in ("travel", "trip", "expense")):
            return [1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def test_text_ingestion_and_rejects_unsupported_files() -> None:
    pages = parse_document("notes.txt", b"Alpha\n\nBeta")
    assert len(pages) == 1
    assert "Alpha" in pages[0].text and "Beta" in pages[0].text
    with pytest.raises(DocumentError, match="Unsupported"):
        parse_document("picture.png", b"not really a picture")
    with pytest.raises(DocumentError, match="empty"):
        parse_document("empty.md", b"")
    with pytest.raises(DocumentError, match="UTF-8"):
        parse_document("broken.txt", b"\xff\xfe")


def test_docx_extraction_includes_paragraphs_and_tables() -> None:
    doc = Document()
    doc.add_paragraph("Confidentiality starts with good process.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Field"
    table.cell(0, 1).text = "Value"
    file = BytesIO()
    doc.save(file)
    pages = parse_document("sample.docx", file.getvalue())
    assert "Confidentiality" in pages[0].text
    assert "Field | Value" in pages[0].text


def test_chunking_has_overlap_and_preserves_page() -> None:
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_pages([PageText(text, page=3)], chunk_words=100, overlap_words=20)
    assert len(chunks) == 3
    assert chunks[0].page == 3
    assert "w80" in chunks[0].text and "w80" in chunks[1].text
    assert "w249" in chunks[-1].text


def test_chunking_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        chunk_pages([PageText("hello")], chunk_words=10)
    with pytest.raises(ValueError):
        chunk_pages([PageText("hello")], chunk_words=20, overlap_words=20)


def test_ingest_retrieve_answer_and_cite_local_passages(tmp_path: Path) -> None:
    settings = Settings(db_path=tmp_path / "qa.sqlite3")
    qa = DocumentQA(settings=settings, embedder=FakeEmbedder())
    content = b"Travel expenses are filed within ten days. A manager approves business travel."
    result = qa.ingest("travel.txt", content)
    assert result["status"] == "indexed"
    assert result["chunks"] == 1
    answer = qa.ask("When should travel expenses be filed?")
    assert answer["answer_mode"] == "extractive"
    assert "[S1]" in answer["answer"]
    assert answer["sources"][0]["filename"] == "travel.txt"
    assert qa.status()["documents"] == 1
    assert qa.status()["chunks"] == 1


def test_duplicate_upload_is_idempotent_and_same_name_replaces(tmp_path: Path) -> None:
    qa = DocumentQA(Settings(db_path=tmp_path / "qa.sqlite3"), embedder=FakeEmbedder())
    first = qa.ingest("policy.md", b"Travel claims are due within ten days.")
    duplicate = qa.ingest("policy.md", b"Travel claims are due within ten days.")
    assert duplicate["status"] == "already_indexed"
    assert qa.status()["documents"] == 1
    second = qa.ingest("policy.md", b"Security credentials are rotated after exposure.")
    assert second["status"] == "indexed"
    docs = qa.store.list_documents()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == second["doc_id"] != first["doc_id"]
    result = qa.ask("What happens to exposed credentials?")
    assert result["sources"][0]["text"].startswith("Security")


def test_model_change_never_compares_incompatible_dense_vectors(tmp_path: Path) -> None:
    qa = DocumentQA(Settings(db_path=tmp_path / "qa.sqlite3"), embedder=FakeEmbedder())
    qa.ingest("travel.txt", b"Travel expenses are filed within ten days.")
    qa.ingest("security.txt", b"Security credentials are rotated after exposure.")

    class NewModel:
        model_name = "different-model-same-dimensions"

        def embed_query(self, text: str) -> list[float]:
            return [0.0, 1.0, 0.0]

    sources = HybridRetriever(qa.store, NewModel()).retrieve("travel expenses", top_k=1)
    assert sources[0].filename == "travel.txt"


def test_filename_is_reduced_to_safe_basename(tmp_path: Path) -> None:
    qa = DocumentQA(Settings(db_path=tmp_path / "qa.sqlite3"), embedder=FakeEmbedder())
    result = qa.ingest("../../plan final?.txt", b"A short travel plan.")
    assert "/" not in result["filename"]
    assert "?" not in result["filename"]


def test_delete_removes_document_and_fts_rows(tmp_path: Path) -> None:
    qa = DocumentQA(Settings(db_path=tmp_path / "qa.sqlite3"), embedder=FakeEmbedder())
    item = qa.ingest("travel.md", b"Travel is planned by the team.")
    assert qa.store.lexical_search("travel")
    assert qa.store.delete_document(item["doc_id"])
    assert qa.store.stats() == {"documents": 0, "chunks": 0}
    assert qa.store.lexical_search("travel") == []
    assert not qa.store.delete_document(item["doc_id"])
