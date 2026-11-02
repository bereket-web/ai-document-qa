from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    db_path: Path
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_file_bytes: int = 15 * 1024 * 1024
    max_files_per_request: int = 10
    chunk_words: int = 180
    overlap_words: int = 35
    top_k: int = 5
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        db_path = Path(os.getenv("DOCQA_DB_PATH", str(PROJECT_ROOT / "data" / "document_qa.sqlite3")))
        return cls(
            db_path=db_path,
            embedding_model=os.getenv(
                "DOCQA_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
            max_file_bytes=int(os.getenv("DOCQA_MAX_FILE_BYTES", 15 * 1024 * 1024)),
            max_files_per_request=int(os.getenv("DOCQA_MAX_FILES", 10)),
            chunk_words=int(os.getenv("DOCQA_CHUNK_WORDS", 180)),
            overlap_words=int(os.getenv("DOCQA_OVERLAP_WORDS", 35)),
            top_k=int(os.getenv("DOCQA_TOP_K", 5)),
            llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            llm_base_url=os.getenv("OPENAI_BASE_URL") or None,
            llm_api_key=os.getenv("OPENAI_API_KEY") or None,
        )
