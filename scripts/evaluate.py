from __future__ import annotations

import json
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docqa.config import Settings  # noqa: E402
from docqa.retrieval import cosine_similarity  # noqa: E402
from docqa.service import DocumentQA  # noqa: E402

METHODS = ("dense_cosine", "bm25", "hybrid_rrf")


def _unique_documents(chunk_ids: list[str], filenames: dict[str, str]) -> list[str]:
    ordered: list[str] = []
    for chunk_id in chunk_ids:
        filename = filenames.get(chunk_id)
        if filename and filename not in ordered:
            ordered.append(filename)
    return ordered


def _metric_row(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    first_rank = next((rank for rank, name in enumerate(ranked, start=1) if name in relevant), None)
    return {
        "hit_at_1": float(bool(ranked and ranked[0] in relevant)),
        "recall_at_3": len(set(ranked[:3]) & relevant) / len(relevant),
        "recall_at_5": len(set(ranked[:5]) & relevant) / len(relevant),
        "reciprocal_rank_at_5": 1 / first_rank if first_rank and first_rank <= 5 else 0.0,
    }


def run_evaluation() -> dict:
    cases = [
        json.loads(line)
        for line in (ROOT / "evaluation" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="docqa-eval-") as tmp:
        settings = Settings(db_path=Path(tmp) / "eval.sqlite3")
        qa = DocumentQA(settings=settings)
        for path in sorted((ROOT / "data" / "demo_documents").glob("*.md")):
            qa.ingest(path.name, path.read_bytes())
        chunks = qa.store.all_chunks(model_name=qa.settings.embedding_model)
        filenames = {chunk.chunk_id: chunk.filename for chunk in chunks}

        for case in cases:
            question = case["question"]
            query_vector = qa.embedder.embed_query(question)
            dense_ids = [
                chunk.chunk_id
                for _, chunk in sorted(
                    ((cosine_similarity(query_vector, chunk.embedding), chunk) for chunk in chunks),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
            ]
            bm25_ids = qa.store.lexical_search(question, limit=max(20, len(chunks)))
            hybrid_ids = [
                hit.chunk_id for hit in qa.retriever.retrieve(question, top_k=max(5, len(chunks)), candidate_k=len(chunks))
            ]
            rankings = {
                "dense_cosine": _unique_documents(dense_ids, filenames),
                "bm25": _unique_documents(bm25_ids, filenames),
                "hybrid_rrf": _unique_documents(hybrid_ids, filenames),
            }
            expected = set(case["relevant_documents"])
            results.append({
                "question": question,
                "relevant_documents": sorted(expected),
                "ranked_documents": rankings,
                "metrics": {method: _metric_row(ranked, expected) for method, ranked in rankings.items()},
            })
        stats = qa.store.stats()

    aggregate = {
        method: {
            metric: mean(row["metrics"][method][metric] for row in results)
            for metric in ("hit_at_1", "recall_at_3", "recall_at_5", "reciprocal_rank_at_5")
        }
        for method in METHODS
    }
    try:
        fastembed_version = version("fastembed")
    except PackageNotFoundError:
        fastembed_version = "unknown"
    summary = {
        "benchmark": "Synthetic internal-policy retrieval smoke benchmark",
        "synthetic_only": True,
        "notes": (
            "Four fictional documents and ten hand-authored, document-level relevance labels; "
            "not a real-user or external QA quality benchmark. One Amharic-language case is a smoke test only."
        ),
        "embedding_model": settings.embedding_model,
        "fastembed_version": fastembed_version,
        "documents": stats["documents"],
        "chunks": stats["chunks"],
        "queries": len(results),
        "metrics_by_retriever": aggregate,
        "per_query": results,
    }
    return summary


if __name__ == "__main__":
    result = run_evaluation()
    output = ROOT / "reports" / "synthetic_retrieval_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"}, ensure_ascii=False, indent=2))
    print(f"Detailed per-query results: {output}")
