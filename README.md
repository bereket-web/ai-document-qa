# Groundwork: AI Document Q&A

A local-first retrieval-augmented generation (RAG) demo for asking questions about uploaded documents. It parses PDF, DOCX, Markdown and plain text; chunks with overlap; creates multilingual embeddings locally; searches with dense cosine similarity plus SQLite FTS5 BM25; fuses ranks; and returns answers with citations to retrieved source passages.

**Default mode needs no API key.** It returns short, verbatim evidence excerpts with citation IDs. Optional OpenAI-compatible generation can be enabled through environment variables; if enabled, retrieved passages are sent to that endpoint. The app is not deployed or published.

## Try the demo

Python 3.10+ is required. On first indexing, FastEmbed downloads the configured multilingual MiniLM model (about 240 MB in this build) from Hugging Face; after that, embeddings run locally using ONNX Runtime.

```bash
uv sync --locked --extra test
uv run python app.py
```

Alternatively, use pip with `python -m venv .venv`, `source .venv/bin/activate`, then `python -m pip install -e '.[test]'` and `python app.py`. The committed `uv.lock` pins the full resolved dependency graph for repeatable `uv` installs.

Open <http://127.0.0.1:7860>, choose **Load demo library**, then try “How long after returning from a trip do employees have to file expenses?” The fictional sample library also includes an Amharic travel-policy example. Upload your own `.pdf`, `.docx`, `.txt`, or `.md` files and ask questions about them. Scanned/image-only PDFs are not OCR'd and are rejected when no text can be extracted.

Run the server on a different interface only in a trusted environment. There is no authentication, so do not expose the app to an untrusted network. The SQLite library is stored in `data/document_qa.sqlite3` by default; set `DOCQA_DB_PATH` to choose another location. Use the “×” controls in the document list or the clear endpoint to remove indexed content.

## Optional LLM generation

Without a key, answers stay extractive and display passages as quotations with their source IDs. To opt in to generated answers, install the LLM extra and configure a compatible endpoint in your shell (do not commit secrets or a populated `.env`):

```bash
python -m pip install -e '.[llm]'
export OPENAI_API_KEY='your-key'
export OPENAI_BASE_URL='https://api.openai.com/v1'  # optional for OpenAI-compatible endpoints
export OPENAI_MODEL='gpt-4o-mini'
python app.py
```

This sends the question and retrieved document excerpts to the configured provider. Choose an endpoint and documents whose data-handling terms are acceptable. The prompt asks the model to use only source passages; returned source IDs are restricted to retrieved IDs, and an answer with no valid citation falls back to a local extractive response. This is a practical safeguard, not a proof that every generated claim is entailed by its cited text.

## System design

- **Ingestion:** local `pypdf` / `python-docx` parsers, UTF-8 text validation, extension allow-list, 15 MB per-file limit, and clear rejection for empty or image-only PDFs.
- **Chunking:** 180-word passages with 35-word overlap by default. PDF page metadata is retained; tune with `DOCQA_CHUNK_WORDS` and `DOCQA_OVERLAP_WORDS`.
- **Embeddings:** [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) through FastEmbed/ONNX Runtime (FastEmbed describes it as covering about 50 languages; the upstream model is Apache-2.0). Vectors are L2-normalized; no E5 prefixes are added to this model.
- **Vector store:** SQLite persists float32 vectors and metadata. Dense retrieval scans the indexed vectors exactly and ranks by cosine similarity. This avoids a separate database service and is intended for a small demo corpus, not millions of chunks.
- **Hybrid retrieval:** dense and FTS5 BM25 rankings are combined with reciprocal-rank fusion (RRF). The fused ranking is the lightweight second-stage reranker; no separate cross-encoder is loaded by default, keeping CPU and memory costs low.
- **Citations:** each retrieved passage gets a request-local label (`[S1]`, `[S2]`, …) and appears in the UI with its file, PDF page when available, and full excerpt. The labels are attached to evidence returned from the store, not constructed by the LLM.
- **Generation:** cited extractive output is the zero-key default; optional OpenAI-compatible chat completion is configured via environment only.
- **Evaluation:** synthetic internal-policy documents with hand-authored, document-level labels; `Hit@1`, `Recall@3`, `Recall@5`, and `MRR@5` are reported side by side for dense cosine, BM25, and fused RRF retrieval, without using an LLM.

Latest synthetic run: 4 fictional documents, 7 passages, and 10 hand-authored questions. Dense cosine scored `Hit@1 = 0.90`, `Recall@3 = 1.00`, `MRR@5 = 0.95`; BM25 and hybrid RRF each scored `Hit@1 = 1.00`, `Recall@3 = 1.00`, `MRR@5 = 1.00`. These are smoke-test counts on a tiny, non-independent sample—not evidence of general retrieval quality. The run used FastEmbed 0.7.4; rerun the suite when changing the embedding model or backend.

See [the architecture diagram](docs/architecture.png) and source [diagram definition](docs/architecture.mmd).

## Tests and evaluation

```bash
pytest
ruff check .
python scripts/evaluate.py
```

The evaluation suite contains four fictional demo documents and ten hand-written questions, including one Amharic question. FastEmbed 0.7.4 also warns that this model mapping uses mean pooling rather than a legacy CLS representation; the report records the backend version. The Amharic case tests the local path but is not evidence of broad language quality. Metrics are an engineering smoke check—not a representative retrieval benchmark, statistical claim, or measure of answer correctness. The detailed run record is written to `reports/synthetic_retrieval_evaluation.json`.

## API

- `GET /api/status` — model, backend and index state.
- `GET /api/documents` — indexed file metadata.
- `POST /api/upload` — multipart document upload.
- `POST /api/demo/load` — index included fictional examples.
- `POST /api/ask` — `{"question":"...","top_k":5}`.
- `DELETE /api/documents/{doc_id}` — remove one file and its passages.
- `DELETE /api/documents` — clear the local library.

Interactive API documentation is available at `/docs` while the app is running.

## Privacy, safety and limitations

Parsing and local embedding do not require document content to leave the machine. The optional generator sends the query and retrieved excerpts to the provider you configure. This app has no authentication and should remain local or behind a trusted access boundary. It is a demo, not an enterprise document-control or access-control product: everyone with access to the running instance can query the whole local library. SQLite exact search is appropriate only for modest collections. PDF extraction quality depends on the source text layer; OCR, tables/layout reconstruction, per-user ACLs, deletion audits, encryption-at-rest, streaming answers, cross-encoder reranking and production-grade observability are not implemented.

Source code is MIT licensed. Demo documents and benchmark labels are original fictional examples. No uploaded files or user documents are included in the repository.
