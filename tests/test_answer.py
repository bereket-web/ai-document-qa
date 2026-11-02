from __future__ import annotations

import sys
from types import SimpleNamespace

from docqa.answer import generate_answer
from docqa.config import Settings
from docqa.retrieval import RetrievedSource


def source() -> RetrievedSource:
    return RetrievedSource(
        source_id="S1", chunk_id="c1", doc_id="d1", filename="policy.md",
        text="The annual budget is USD 1,200.", page=None, chunk_index=0, score=0.03, metadata={},
    )


def test_llm_citations_are_restricted_to_retrieved_sources(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            message = SimpleNamespace(content="The budget is USD 1,200 [S1]. Unsupported [S99].")
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))
    settings = Settings(db_path=__import__("pathlib").Path("unused.sqlite3"), llm_api_key="test-key")
    result = generate_answer("What is the budget?", [source()], settings)
    assert result.mode == "llm"
    assert "[S1]" in result.answer
    assert "S99" not in result.answer


def test_llm_without_a_valid_source_id_falls_back(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            message = SimpleNamespace(content="Probably USD 1,200 [S99].")
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))
    settings = Settings(db_path=__import__("pathlib").Path("unused.sqlite3"), llm_api_key="test-key")
    result = generate_answer("What is the budget?", [source()], settings)
    assert result.mode == "extractive_fallback"
    assert "The annual budget is USD 1,200" in result.answer
    assert "[S1]" in result.answer
