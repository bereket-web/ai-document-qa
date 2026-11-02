from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class DocumentError(ValueError):
    """Raised when a document cannot be safely parsed."""


@dataclass(frozen=True)
class PageText:
    text: str
    page: int | None = None


def parse_document(filename: str, content: bytes, max_bytes: int = 15 * 1024 * 1024) -> list[PageText]:
    """Extract text locally. No uploaded content is sent to an external parser."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentError(f"Unsupported file type {extension or '(none)'}. Use PDF, DOCX, TXT, or MD.")
    if not content:
        raise DocumentError("The uploaded file is empty.")
    if len(content) > max_bytes:
        raise DocumentError(f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")

    try:
        if extension == ".pdf":
            reader = PdfReader(io.BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise DocumentError("Password-protected PDFs are not supported.")
            pages = [PageText(text=(page.extract_text() or ""), page=i + 1) for i, page in enumerate(reader.pages)]
        elif extension == ".docx":
            doc = Document(io.BytesIO(content))
            lines = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    if any(cells):
                        lines.append(" | ".join(cells))
            pages = [PageText(text="\n".join(lines), page=None)]
        else:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise DocumentError("Text files must be UTF-8 encoded.") from exc
            pages = [PageText(text=text, page=None)]
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Could not parse this {extension[1:].upper()} file: {exc.__class__.__name__}.") from exc

    cleaned: list[PageText] = []
    for item in pages:
        text = re.sub(r"[\t\r ]+", " ", item.text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            cleaned.append(PageText(text=text, page=item.page))
    if not cleaned:
        raise DocumentError("No extractable text was found. Scanned/image-only PDFs need OCR before indexing.")
    return cleaned
