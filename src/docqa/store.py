from __future__ import annotations

import json
import re
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    page: int | None
    chunk_index: int
    embedding: list[float]
    model_name: str
    metadata: dict[str, Any]


class VectorStore:
    """Small-corpus exact cosine vector store plus SQLite FTS5 lexical index."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page INTEGER,
                    embedding BLOB NOT NULL,
                    embedding_dims INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, chunk_index);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED, text, source_name, tokenize='unicode61 remove_diacritics 0'
                );
                """
            )

    @staticmethod
    def _pack(vector: list[float]) -> bytes:
        return struct.pack("<" + "f" * len(vector), *vector)

    @staticmethod
    def _unpack(blob: bytes, dimensions: int) -> list[float]:
        return list(struct.unpack("<" + "f" * dimensions, blob))

    def has_digest(self, filename: str, digest: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM documents WHERE filename=? AND sha256=?", (filename, digest)).fetchone() is not None

    def replace_document(
        self,
        *,
        doc_id: str,
        filename: str,
        digest: str,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        model_name: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Every chunk must have exactly one embedding")
        if not chunks:
            raise ValueError("Cannot index an empty document")
        dims = len(vectors[0])
        if dims == 0 or any(len(vector) != dims for vector in vectors):
            raise ValueError("Embedding dimensions are empty or inconsistent")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            old_ids = [row[0] for row in db.execute("SELECT chunk_id FROM chunks WHERE doc_id IN (SELECT doc_id FROM documents WHERE filename=?)", (filename,))]
            if old_ids:
                db.executemany("DELETE FROM chunks_fts WHERE chunk_id=?", [(chunk_id,) for chunk_id in old_ids])
            db.execute("DELETE FROM documents WHERE filename=?", (filename,))
            db.execute("INSERT INTO documents(doc_id,filename,sha256,indexed_at,chunk_count) VALUES(?,?,?,?,?)", (doc_id, filename, digest, now, len(chunks)))
            for item, vector in zip(chunks, vectors, strict=True):
                chunk_id = item["chunk_id"]
                metadata = dict(item.get("metadata", {}))
                db.execute(
                    "INSERT INTO chunks(chunk_id,doc_id,chunk_index,text,page,embedding,embedding_dims,model_name,metadata) VALUES(?,?,?,?,?,?,?,?,?)",
                    (chunk_id, doc_id, int(item["chunk_index"]), item["text"], item.get("page"), self._pack(vector), dims, model_name, json.dumps(metadata, ensure_ascii=False)),
                )
                db.execute("INSERT INTO chunks_fts(chunk_id,text,source_name) VALUES(?,?,?)", (chunk_id, item["text"], filename))

    def all_chunks(self, model_name: str | None = None) -> list[StoredChunk]:
        with self._connect() as db:
            sql = "SELECT c.*, d.filename FROM chunks c JOIN documents d USING(doc_id)"
            params: tuple[Any, ...] = ()
            if model_name:
                sql += " WHERE c.model_name=?"
                params = (model_name,)
            rows = db.execute(sql, params).fetchall()
        return [StoredChunk(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"], filename=row["filename"], text=row["text"],
            page=row["page"], chunk_index=row["chunk_index"], embedding=self._unpack(row["embedding"], row["embedding_dims"]),
            model_name=row["model_name"], metadata=json.loads(row["metadata"]),
        ) for row in rows]

    def lexical_search(self, query: str, limit: int = 20) -> list[str]:
        tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not tokens:
            return []
        match_query = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens[:32])
        try:
            with self._connect() as db:
                rows = db.execute(
                    "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                    (match_query, limit),
                ).fetchall()
            return [row[0] for row in rows]
        except sqlite3.OperationalError:
            return []

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT doc_id,filename,sha256,indexed_at,chunk_count FROM documents ORDER BY indexed_at DESC, filename").fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        with self._connect() as db:
            docs = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": docs, "chunks": chunks}

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as db:
            ids = [row[0] for row in db.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))]
            if not ids:
                return False
            db.executemany("DELETE FROM chunks_fts WHERE chunk_id=?", [(item,) for item in ids])
            db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        return True

    def clear(self) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM chunks_fts")
            db.execute("DELETE FROM documents")
