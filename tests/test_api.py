from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module
from docqa.config import Settings
from docqa.service import DocumentQA


class FakeEmbedder:
    model_name = "api-test-embedding"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_status_and_demo_ui_are_available() -> None:
    client = TestClient(app_module.app)
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["vector_backend"].startswith("SQLite exact cosine")
    assert response.json()["embedding_loaded"] is False
    page = client.get("/")
    assert page.status_code == 200
    assert "Groundwork" in page.text
    assert "Load demo library" in page.text


def test_upload_then_question_returns_cited_source(tmp_path: Path, monkeypatch) -> None:
    service = DocumentQA(Settings(db_path=tmp_path / "api.sqlite3"), embedder=FakeEmbedder())
    monkeypatch.setattr(app_module, "qa", service)
    client = TestClient(app_module.app)
    upload = client.post("/api/upload", files={"files": ("travel.txt", b"File expense claims within ten days.", "text/plain")})
    assert upload.status_code == 200
    assert upload.json()["results"][0]["status"] == "indexed"
    response = client.post("/api/ask", json={"question": "When do I file claims?"})
    assert response.status_code == 200
    body = response.json()
    assert "[S1]" in body["answer"]
    assert body["sources"][0]["filename"] == "travel.txt"
