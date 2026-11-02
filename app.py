from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from docqa.config import PROJECT_ROOT
from docqa.service import DocumentQA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("docqa")
qa = DocumentQA()
app = FastAPI(title="Document Q&A", version="0.1.0", description="Local-first, citation-grounded RAG demo")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=8)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "static" / "index.html")


@app.get("/api/status")
def status() -> dict:
    return qa.status()


@app.get("/api/documents")
def documents() -> dict:
    return {"documents": qa.store.list_documents(), **qa.store.stats()}


@app.post("/api/upload")
def upload(files: Annotated[list[UploadFile], File()]) -> dict:
    if len(files) > qa.settings.max_files_per_request:
        raise HTTPException(413, f"Upload at most {qa.settings.max_files_per_request} files at a time.")
    results = []
    for file in files:
        filename = file.filename or "document.txt"
        try:
            content = file.file.read(qa.settings.max_file_bytes + 1)
            result = qa.ingest(filename, content)
            results.append(result)
        except ValueError as exc:
            results.append({"filename": Path(filename).name, "status": "error", "error": str(exc)})
        except Exception as exc:
            logger.exception("Document indexing failed for %s", Path(filename).name)
            results.append({"filename": Path(filename).name, "status": "error", "error": f"Indexing failed ({exc.__class__.__name__})."})
        finally:
            file.file.close()
    return {"results": results, **qa.store.stats()}


@app.post("/api/demo/load")
def load_demo() -> dict:
    folder = PROJECT_ROOT / "data" / "demo_documents"
    results = []
    for path in sorted(folder.glob("*.md")):
        try:
            results.append(qa.ingest(path.name, path.read_bytes()))
        except Exception as exc:
            logger.exception("Could not index demo document %s", path.name)
            results.append({"filename": path.name, "status": "error", "error": f"Indexing failed ({exc.__class__.__name__})."})
    return {"results": results, **qa.store.stats()}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    try:
        return qa.ask(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Question answering failed")
        raise HTTPException(503, f"The question could not be processed ({exc.__class__.__name__}).") from exc


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    deleted = qa.store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(404, "Document not found.")
    return {"deleted": True, **qa.store.stats()}


@app.delete("/api/documents")
def clear_documents() -> dict:
    qa.store.clear()
    return {"cleared": True, **qa.store.stats()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=7860, reload=False)
