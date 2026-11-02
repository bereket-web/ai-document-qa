from __future__ import annotations

import json
from pathlib import Path

from docqa.chunking import chunk_pages
from docqa.ingest import PageText


def test_demo_evaluation_labels_are_well_formed() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = [json.loads(line) for line in (root / "evaluation" / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    demo_docs = {path.name for path in (root / "data" / "demo_documents").glob("*.md")}
    assert len(cases) >= 8
    assert any(any("\u1200" <= char <= "\u137f" for char in case["question"]) for case in cases)
    assert all(case["relevant_documents"] and set(case["relevant_documents"]) <= demo_docs for case in cases)


def test_page_chunks_are_nonempty_and_contain_source_text() -> None:
    chunks = chunk_pages([PageText("A useful source passage for grounded retrieval.", 1)], chunk_words=40, overlap_words=5)
    assert len(chunks) == 1
    assert "grounded retrieval" in chunks[0].text
