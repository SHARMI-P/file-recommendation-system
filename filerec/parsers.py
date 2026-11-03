"""Turn files into text.

Default parsers are lightweight and fast on a CPU laptop (pypdf, python-docx, python-pptx, openpyxl).
Set FILEREC_PARSER=docling to use Docling for higher-quality layout-aware parsing of PDF/DOCX/PPTX
(slower, downloads layout models on first use). Images and scanned PDFs use Tesseract OCR when
FILEREC_USE_OCR=1 and Tesseract is installed.
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from .config import SETTINGS

log = logging.getLogger(__name__)
PARSER = os.getenv("FILEREC_PARSER", "light")
_docling = None


def _docling_convert(path: Path) -> str | None:
    global _docling
    try:
        if _docling is None:
            from docling.document_converter import DocumentConverter
            _docling = DocumentConverter()
        return _docling.convert(str(path)).document.export_to_markdown()
    except Exception as e:  # noqa: BLE001
        log.warning("Docling failed on %s (%s); falling back", path.name, e)
        return None


def _ocr_image(path: Path) -> str:
    if not SETTINGS.use_ocr:
        return ""
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(path))
    except Exception as e:  # noqa: BLE001
        log.warning("OCR failed on %s: %s", path.name, e)
        return ""


def _pdf(path: Path) -> tuple[str, dict]:
    from pypdf import PdfReader
    r = PdfReader(str(path))
    text = "\n".join((p.extract_text() or "") for p in r.pages[:80])
    meta = {}
    if r.metadata:
        meta = {"title": r.metadata.get("/Title"), "author": r.metadata.get("/Author")}
    if len(text.strip()) < 30 and SETTINGS.use_ocr:  # probably a scan
        try:
            import pytesseract
            from pdf2image import convert_from_path
            text = "\n".join(pytesseract.image_to_string(im) for im in convert_from_path(str(path), last_page=10))
        except Exception as e:  # noqa: BLE001
            log.warning("Scanned-PDF OCR failed on %s: %s", path.name, e)
    return text, meta


def _docx(path: Path) -> tuple[str, dict]:
    import docx
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text for c in row.cells))
    cp = d.core_properties
    return "\n".join(parts), {"title": cp.title, "author": cp.author}


def _pptx(path: Path) -> tuple[str, dict]:
    from pptx import Presentation
    p = Presentation(str(path))
    parts = []
    for i, slide in enumerate(p.slides, 1):
        texts = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        if slide.has_notes_slide:
            texts.append(slide.notes_slide.notes_text_frame.text)
        parts.append(f"[Slide {i}] " + "\n".join(texts))
    cp = p.core_properties
    return "\n".join(parts), {"title": cp.title, "author": cp.author}


def _xlsx(path: Path) -> tuple[str, dict]:
    import openpyxl
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        parts.append(f"[Sheet {ws.title}]")
        for n, row in enumerate(ws.iter_rows(values_only=True)):
            if n > 300:
                break
            cells = [str(c) for c in row if c is not None]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts), {"title": wb.properties.title, "author": wb.properties.creator}


def _csv(path: Path) -> tuple[str, dict]:
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        rows = [" | ".join(r) for _, r in zip(range(300), csv.reader(f))]
    return "\n".join(rows), {}


def _text(path: Path) -> tuple[str, dict]:
    return path.read_text(encoding="utf-8", errors="ignore")[:200_000], {}


def parse(path: Path) -> tuple[str, dict]:
    """Return (text, metadata). Text may be empty for images without OCR; the filename is still indexed."""
    ext = path.suffix.lower()
    try:
        if PARSER == "docling" and ext in {".pdf", ".docx", ".pptx", ".html"}:
            md = _docling_convert(path)
            if md:
                return md, {}
        if ext == ".pdf":
            return _pdf(path)
        if ext == ".docx":
            return _docx(path)
        if ext == ".pptx":
            return _pptx(path)
        if ext == ".xlsx":
            return _xlsx(path)
        if ext == ".csv":
            return _csv(path)
        if ext in {".png", ".jpg", ".jpeg"}:
            return _ocr_image(path), {}
        return _text(path)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not parse %s: %s", path, e)
        return "", {}


def chunk(text: str, size: int, overlap: int, limit: int) -> list[str]:
    """Paragraph-aware character chunking."""
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:  # very long paragraph
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:size])
            p = p[size - overlap:]
        if len(cur) + len(p) + 1 > size and cur:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n" + p
        else:
            cur = f"{cur}\n{p}" if cur else p
        if len(chunks) >= limit:
            break
    if cur and len(chunks) < limit:
        chunks.append(cur)
    return chunks[:limit]
