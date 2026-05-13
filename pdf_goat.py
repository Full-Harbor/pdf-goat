#!/usr/bin/env python3
"""pdf_goat.py — Canonical PDF processor for the Harbor Ingest pipeline.

The ONE way PDFs enter the corpus. No other script should import pdfplumber,
pdf2image, or call Claude Vision directly. If you need PDF content, call
pdf_goat.

Pipeline (12 steps):
  1. Hash + dedupe (SHA256 — skip if already in manifest)
  2. Text extraction (pdfplumber)
  3. Page classification (text_only / text_with_images / image_only / empty)
  4. Table extraction (pdfplumber tables)
  5. Vision routing (image_only → full_vision, mixed → page_vision)
  6. Claude Vision extraction (for flagged pages)
  7. Merge text + vision output per page
  8. Preprocess (chunking.preprocess_text)
  9. Chunk (chunking.chunk_text — v2 recursive)
 10. Embed (text-embedding-3-large @ 1536d)
 11. Optional upsert to Supabase (configurable table, on_conflict)
 12. Write manifest row (local CSV)

Usage:
  # Single file
  python pdf_goat.py /path/to/doc.pdf

  # Directory (process all PDFs)
  python pdf_goat.py /path/to/staging/

  # Dry run (no Supabase writes, no embeddings — just extract + manifest)
  python pdf_goat.py /path/to/doc.pdf --dry-run

  # Force reprocess (ignore manifest / dedupe)
  python pdf_goat.py /path/to/doc.pdf --force

  # Skip vision (text-only extraction, no API calls)
  python pdf_goat.py /path/to/doc.pdf --no-vision

  # Vision-only (only process pages flagged as image_only)
  python pdf_goat.py /path/to/doc.pdf --vision-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Fix PATH for Homebrew on macOS (poppler, tesseract)
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    os.environ.setdefault(
        "PATH",
        "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + os.environ.get("PATH", ""),
    )

# Avoid PIL picking up wrong Python version
sys.path = [p for p in sys.path if "python3.13" not in p]

import pdfplumber
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project imports (sibling modules)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from chunking import chunk_text, preprocess_text
from config import (
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    CHUNKER_VERSION,
    EMBED_DIMENSIONS,
    EMBEDDING_MODEL,
    MAX_RETRIES,
    RETRY_BASE_WAIT,
    IMPORT_BATCH_VERSION,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pdf_goat")

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
ENV_PATH = SCRIPT_DIR.parent / ".env"
load_dotenv(ENV_PATH)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MANIFEST_PATH = SCRIPT_DIR.parent / "audits" / "pdf_goat_manifest.csv"
MANIFEST_FIELDS = [
    "sha256", "filename", "path", "page_count", "total_text_chars",
    "vision_pages", "chunks_produced", "embedded_at", "chunk_type",
    "embedding_model", "chunker_version", "vision_prompt_template",
    "pdf_goat_version", "error",
]

PDF_GOAT_VERSION = "1.0.0"  # bump on any pipeline logic change

VISION_MODEL = "claude-sonnet-4-20250514"
VISION_MAX_TOKENS = 4096

# Named prompt templates for vision extraction
VISION_PROMPTS: dict[str, str] = {
    "ar_metrics_8col": (
        "This table has EXACTLY 8 columns: "
        "JAN · CANCELLED · FEB · CANCELLED · YTD · # to meet goal · 2026 GOAL · % complete. "
        "Extract every row preserving all 8 columns. Use — for blank cells. "
        "Output as a markdown table."
    ),
    "990_form": (
        "This is an IRS Form 990 or 990-PF page. Extract all field labels and their values. "
        "Preserve Part/Schedule/Line structure. Output as structured text with "
        "Part > Line > Value format."
    ),
    "general": (
        "Extract all text content from this page. Preserve headings, paragraphs, "
        "bullet points, and table structures. Output as markdown."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Hash + Dedupe
# ═══════════════════════════════════════════════════════════════════════════

def file_sha256(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict[str, dict]:
    """Load existing manifest as {sha256: row_dict}."""
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, newline="") as f:
        return {row["sha256"]: row for row in csv.DictReader(f)}


def append_manifest(row: dict) -> None:
    """Append a single row to the manifest CSV."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
# Step 2-4: Text Extraction + Page Classification + Tables
# ═══════════════════════════════════════════════════════════════════════════

class PageInfo:
    """Holds extraction results for a single PDF page."""
    __slots__ = (
        "page_num", "text", "images", "tables", "classification",
        "vision_text", "merged_text",
    )

    def __init__(self, page_num: int):
        self.page_num = page_num
        self.text: str = ""
        self.images: list = []
        self.tables: list[list[list[str]]] = []
        self.classification: str = ""  # text_only|text_with_images|image_only|empty
        self.vision_text: str = ""
        self.merged_text: str = ""


def extract_pages(pdf_path: Path) -> list[PageInfo]:
    """Extract text, images, tables from every page. Classify each page."""
    pages: list[PageInfo] = []
    # File-level byte-per-page estimate: reliable proxy for raster content.
    # A scanned letter page at 200dpi is ~100-300 KB; a truly blank page is <1 KB.
    try:
        file_bytes = pdf_path.stat().st_size
    except OSError:
        file_bytes = 0

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = max(len(pdf.pages), 1)
        bytes_per_page_avg = file_bytes / n_pages

        for i, page in enumerate(pdf.pages):
            pi = PageInfo(page_num=i + 1)

            # Text
            raw = page.extract_text() or ""
            pi.text = raw.strip()

            # Images
            pi.images = page.images or []

            # Tables
            try:
                tbls = page.extract_tables() or []
                pi.tables = tbls
            except Exception:
                pi.tables = []

            # Classify
            has_text = len(pi.text) > 20
            has_images = len(pi.images) > 0

            if has_text and has_images:
                pi.classification = "text_with_images"
            elif has_text:
                pi.classification = "text_only"
            elif has_images:
                pi.classification = "image_only"
            else:
                # Distinguish truly-blank pages from scanned raster pages.
                # pdfplumber can't enumerate inline images on raster page streams
                # (the page IS the image), so those falsely show 0 images.
                # Heuristic: if the average bytes/page is large the file almost
                # certainly contains scanned raster data, not blank pages.
                # Threshold: 50 KB/page — a blank page is <1 KB, scanned is >80 KB.
                if bytes_per_page_avg > 50_000:
                    pi.classification = "image_only"  # scanned raster page
                else:
                    pi.classification = "empty"

            pages.append(pi)
    return pages


def tables_to_markdown(tables: list[list[list[str]]]) -> str:
    """Convert pdfplumber table data to markdown tables."""
    parts = []
    for tbl in tables:
        if not tbl or not tbl[0]:
            continue
        # Header row
        header = [str(c or "").strip() for c in tbl[0]]
        parts.append("| " + " | ".join(header) + " |")
        parts.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in tbl[1:]:
            cells = [str(c or "").strip() for c in row]
            # Pad if needed
            while len(cells) < len(header):
                cells.append("")
            parts.append("| " + " | ".join(cells[:len(header)]) + " |")
        parts.append("")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Step 5-6: Vision Routing + Claude Vision
# ═══════════════════════════════════════════════════════════════════════════

def detect_prompt_template(filename: str, page_num: int) -> str:
    """Pick the right vision prompt template based on filename patterns."""
    fn_lower = filename.lower()

    # Association Reports — metrics tables
    if any(kw in fn_lower for kw in ["association-report", "association_report", "-ar-"]):
        return "ar_metrics_8col"

    # 990 tax filings
    if "990" in fn_lower:
        return "990_form"

    return "general"


def render_page_to_base64(pdf_path: Path, page_num: int) -> str:
    """Render a single PDF page to base64 PNG using pdf2image."""
    from pdf2image import convert_from_path  # lazy import — requires poppler

    images = convert_from_path(
        str(pdf_path),
        first_page=page_num,
        last_page=page_num,
        dpi=200,
        fmt="png",
    )
    if not images:
        return ""

    import base64
    from io import BytesIO

    buf = BytesIO()
    images[0].save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def vision_extract_page(
    pdf_path: Path,
    page_num: int,
    prompt_template: str = "general",
) -> str:
    """Send a single page image to Claude Vision and return extracted text."""
    import httpx

    if not ANTHROPIC_API_KEY:
        log.warning("No ANTHROPIC_API_KEY — skipping vision for page %d", page_num)
        return ""

    b64 = render_page_to_base64(pdf_path, page_num)
    if not b64:
        return ""

    prompt = VISION_PROMPTS.get(prompt_template, VISION_PROMPTS["general"])

    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": VISION_MODEL,
                    "max_tokens": VISION_MAX_TOKENS,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }],
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            wait = RETRY_BASE_WAIT * (2 ** attempt)
            log.warning(
                "Vision API error page %d attempt %d/%d: %s — retrying in %.1fs",
                page_num, attempt + 1, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

    log.error("Vision extraction failed after %d retries for page %d", MAX_RETRIES, page_num)
    return ""


def run_vision_pass(
    pdf_path: Path,
    pages: list[PageInfo],
    *,
    vision_only: bool = False,
) -> int:
    """Run Claude Vision on pages that need it. Returns count of pages processed."""
    vision_pages = [
        p for p in pages
        if p.classification == "image_only"
        or (p.classification == "empty" and len(p.images) > 0)
        or (vision_only and p.classification == "empty")  # scanned pages: no inline imgs
    ]

    if not vision_pages:
        log.info("No vision-needed pages detected")
        return 0

    log.info("Vision pass: %d pages to process", len(vision_pages))
    filename = pdf_path.name

    for pi in vision_pages:
        template = detect_prompt_template(filename, pi.page_num)
        log.info(
            "  Page %d → template=%s", pi.page_num, template,
        )
        pi.vision_text = vision_extract_page(pdf_path, pi.page_num, template)
        if pi.vision_text:
            log.info("    ✓ extracted %d chars", len(pi.vision_text))

    return len(vision_pages)


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Merge
# ═══════════════════════════════════════════════════════════════════════════

def merge_page_content(pages: list[PageInfo]) -> str:
    """Merge text + vision + tables into a single document string."""
    parts = []
    for pi in pages:
        page_parts = []

        # Vision text takes priority for image_only pages
        if pi.vision_text:
            page_parts.append(pi.vision_text)
        elif pi.text:
            page_parts.append(pi.text)

        # Append table markdown if tables were detected via pdfplumber
        if pi.tables and not pi.vision_text:
            tbl_md = tables_to_markdown(pi.tables)
            if tbl_md.strip():
                page_parts.append(tbl_md)

        if page_parts:
            header = f"\n\n--- Page {pi.page_num} ---\n\n"
            parts.append(header + "\n\n".join(page_parts))

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Step 8-9: Preprocess + Chunk
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_and_chunk(full_text: str) -> list[str]:
    """Clean text and split into chunks."""
    cleaned = preprocess_text(full_text, is_ocr=False)
    if not cleaned:
        return []
    return chunk_text(cleaned, chunk_size=CHUNK_CHARS, overlap=CHUNK_OVERLAP)


# ═══════════════════════════════════════════════════════════════════════════
# Step 10: Embed
# ═══════════════════════════════════════════════════════════════════════════

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed chunks via OpenAI text-embedding-3-large @ 1536d."""
    import openai

    if not OPENAI_API_KEY:
        log.warning("No OPENAI_API_KEY — skipping embedding")
        return []

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    embeddings: list[list[float]] = []

    # Batch in groups of 100 (API limit is 2048 but smaller is safer)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimensions=EMBED_DIMENSIONS,
                )
                embeddings.extend([d.embedding for d in resp.data])
                break
            except Exception as exc:
                wait = RETRY_BASE_WAIT * (2 ** attempt)
                log.warning(
                    "Embed error batch %d attempt %d/%d: %s — retrying in %.1fs",
                    i // batch_size, attempt + 1, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        else:
            log.error("Embedding failed after %d retries for batch %d", MAX_RETRIES, i // batch_size)
            embeddings.extend([[] for _ in batch])

    return embeddings


# ═══════════════════════════════════════════════════════════════════════════
# Step 11: Upsert to Supabase
# ═══════════════════════════════════════════════════════════════════════════

def supabase_row_count(sha256: str, table: str = "document_embeddings") -> int:
    """Query actual DB row count for a sha256. Returns -1 if credentials absent."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return -1
    import requests as _req
    r = _req.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Prefer": "count=exact",
        },
        params={"source_pdf_sha256": f"eq.{sha256}", "select": "id", "limit": 0},
        timeout=15,
    )
    cr = r.headers.get("content-range", "")
    try:
        return int(cr.split("/")[-1])
    except (ValueError, IndexError):
        return -1


def upsert_to_supabase(
    rows: list[dict[str, Any]],
    table: str = "document_embeddings",
) -> int:
    """Upsert rows to Supabase. Returns count of rows written.

    Raises RuntimeError if any batch receives a non-2xx response — upsert
    failures are surfaced as errors, not silently swallowed.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        log.warning("No Supabase credentials — skipping upsert")
        return 0

    from supabase import create_client

    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    # Batch upsert in groups of 500
    written = 0
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        try:
            client.table(table).upsert(
                batch,
                on_conflict="object_id,chunk_type,row_ordinal",
            ).execute()
            written += len(batch)
        except Exception as exc:
            # Surface as a hard error so batch runners count this doc as failed.
            raise RuntimeError(
                f"Supabase upsert failed on batch starting at index {i}: {exc}"
            ) from exc

    return written


def build_supabase_rows(
    filename: str,
    sha256: str,
    chunks: list[str],
    embeddings: list[list[float]],
    chunk_type: str,
    corpus: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build row dicts for a configurable Supabase embeddings table."""
    now = datetime.now(timezone.utc).isoformat()
    corpus_tag = corpus if corpus is not None else ["public_docs"]
    rows = []
    for ordinal, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "object_id": sha256,
            "filer_name": os.path.splitext(filename)[0],
            "chunk_type": chunk_type,
            "row_ordinal": ordinal,
            "chunk_text": chunk.replace("\x00", ""),
            "embedding": emb if emb else None,
            "embedding_model": EMBEDDING_MODEL,
            "embedded_at": now,
            "import_batch": IMPORT_BATCH_VERSION,
            "source_xml_sha256": sha256,
            "source_pdf_sha256": sha256,
            "corpus": corpus_tag,
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Step 12: Classify chunk_type from filename
# ═══════════════════════════════════════════════════════════════════════════

def classify_chunk_type(filename: str, chunk_prefix: str = "public") -> str:
    """Infer chunk_type from filename patterns. chunk_prefix overrides the org prefix."""
    fn = filename.lower()
    p = chunk_prefix

    if "990" in fn:
        if "foundation" in fn:
            return f"{p}_tier1_foundation"
        return f"{p}_tier1_990_tax"

    if "audit" in fn:
        return f"{p}_tier1_financial_audit"

    if "association-report" in fn or "association_report" in fn:
        return f"{p}_tier1_reports_strategy"

    if "strategic" in fn or "annual-report" in fn or "annual_report" in fn:
        return f"{p}_tier1_reports_strategy"

    if "minutes" in fn or "bod" in fn:
        return f"{p}_tier2_minutes"

    if "president" in fn:
        return f"{p}_tier1_governance"

    if "safety" in fn or "overboard" in fn or "rescue" in fn:
        return f"{p}_tier3_safety_training"

    if "judge" in fn or "appeal" in fn or "rule" in fn:
        return f"{p}_tier3_rules_appeals"

    if "selection" in fn or "team" in fn or "olympic" in fn:
        return f"{p}_tier3_selection_team"

    if "nor" in fn or "regatta" in fn or "championship" in fn:
        return f"{p}_tier4_regatta_nor"

    if "one-design" in fn or "one_design" in fn:
        return f"{p}_tier4_one_design"

    if "newsletter" in fn or "community" in fn:
        return f"{p}_tier4_community"

    return f"{p}_tier4_other"


# ═══════════════════════════════════════════════════════════════════════════
# Module-level corpus/prefix globals (overridden by CLI args in main())
# ═══════════════════════════════════════════════════════════════════════════
_CORPUS: list[str] = ["public_docs"]
_CHUNK_PREFIX: str = "public"
_TABLE: str = "document_embeddings"


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_path: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    no_vision: bool = False,
    vision_only: bool = False,
) -> dict:
    """Run the full 12-step pipeline on a single PDF. Returns manifest row."""
    filename = pdf_path.name
    log.info("━" * 60)
    log.info("Processing: %s", filename)

    # Step 1: Hash + dedupe
    sha = file_sha256(pdf_path)
    if not force:
        manifest = load_manifest()
        if sha in manifest and not manifest[sha].get("error"):
            mrow = manifest[sha]
            chunks_produced = int(mrow.get("chunks_produced") or 0)
            vp_raw = mrow.get("vision_pages") or 0; vision_pages = len(str(vp_raw).split(",")) if vp_raw and str(vp_raw).strip() not in ("0", "") else int(vp_raw or 0)
            # Ghost guard 1: vision doc that produced 0 chunks → reprocess.
            if vision_pages > 0 and chunks_produced == 0:
                log.info(
                    "  ⚠  Manifest hit but vision_pages=%d and chunks=0 — ghost row, reprocessing (SHA256=%s…).",
                    vision_pages, sha[:12],
                )
            else:
                # Ghost guard 2: manifest claims N chunks but DB has fewer.
                # This catches null-byte / upsert failures that logged "success".
                db_count = supabase_row_count(sha, table=_TABLE)
                if db_count >= 0 and db_count < chunks_produced:
                    log.warning(
                        "  ⚠  Manifest=%d chunks but DB=%d rows for SHA256=%s… — re-embedding.",
                        chunks_produced, db_count, sha[:12],
                    )
                else:
                    log.info("  ⏭  Already in manifest and DB confirmed (SHA256=%s…). Use --force to reprocess.", sha[:12])
                    result = dict(mrow)
                    result["_skipped"] = True
                    return result

    # Step 2-4: Extract
    try:
        pages = extract_pages(pdf_path)
    except Exception as exc:
        log.error("  ✗ Extraction failed: %s", exc)
        row = {k: "" for k in MANIFEST_FIELDS}
        row.update(sha256=sha, filename=filename, path=str(pdf_path), error=str(exc))
        append_manifest(row)
        return row

    total_text = sum(len(p.text) for p in pages)
    vision_needed = [p for p in pages if p.classification == "image_only"]
    log.info(
        "  %d pages | %d chars text | %d image-only pages",
        len(pages), total_text, len(vision_needed),
    )

    # Step 5-6: Vision
    vision_count = 0
    empty_pages = [p for p in pages if p.classification == "empty"]
    if not no_vision and (vision_needed or (vision_only and empty_pages)):
        vision_count = run_vision_pass(pdf_path, pages, vision_only=vision_only)

    # Step 7: Merge
    full_text = merge_page_content(pages)
    if not full_text.strip():
        log.warning("  ⚠ No content extracted")

    # Step 8-9: Chunk
    chunks = preprocess_and_chunk(full_text)
    log.info("  %d chunks produced", len(chunks))

    # Classify
    chunk_type = classify_chunk_type(filename, chunk_prefix=_CHUNK_PREFIX)

    # Step 10-11: Embed + Upsert (skip in dry_run)
    if dry_run:
        log.info("  🏜  Dry run — skipping embed + upsert")
        embeddings = []
    else:
        embeddings = embed_chunks(chunks)
        if embeddings:
            sb_rows = build_supabase_rows(filename, sha, chunks, embeddings, chunk_type, corpus=_CORPUS)
            written = upsert_to_supabase(sb_rows, table=_TABLE)
            log.info("  ✓ Upserted %d rows to Supabase", written)

    # Step 12: Manifest
    vision_page_nums = ",".join(
        str(p.page_num) for p in pages if p.vision_text
    )
    # Detect which vision templates were used
    vision_templates_used = set()
    if not no_vision:
        for p in pages:
            if p.vision_text:
                vision_templates_used.add(detect_prompt_template(filename, p.page_num))

    row = {
        "sha256": sha,
        "filename": filename,
        "path": str(pdf_path),
        "page_count": str(len(pages)),
        "total_text_chars": str(total_text),
        "vision_pages": vision_page_nums,
        "chunks_produced": str(len(chunks)),
        "embedded_at": datetime.now(timezone.utc).isoformat() if not dry_run else "",
        "chunk_type": chunk_type,
        "embedding_model": f"{EMBEDDING_MODEL}@{EMBED_DIMENSIONS}d" if not dry_run else "",
        "chunker_version": CHUNKER_VERSION,
        "vision_prompt_template": ",".join(sorted(vision_templates_used)) if vision_templates_used else "",
        "pdf_goat_version": PDF_GOAT_VERSION,
        "error": "",
    }
    append_manifest(row)
    log.info("  ✓ Done: %s → %d chunks (%s)", filename, len(chunks), chunk_type)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="pdf_goat — Canonical PDF processor for Harbor Ingest",
    )
    parser.add_argument("path", type=Path, help="PDF file or directory of PDFs")
    parser.add_argument("--dry-run", action="store_true", help="Extract only, no embed/upsert")
    parser.add_argument("--force", action="store_true", help="Reprocess even if in manifest")
    parser.add_argument("--no-vision", action="store_true", help="Skip vision extraction")
    parser.add_argument("--vision-only", action="store_true", help="Only process image-only pages")
    parser.add_argument(
        "--corpus", nargs="+", default=None,
        help="Corpus tag(s) for output rows (default: ['public_docs'])",
    )
    parser.add_argument(
        "--chunk-prefix", default=None,
        help="Prefix for chunk_type classification (default: 'public')",
    )
    parser.add_argument(
        "--table", default="document_embeddings",
        help="Supabase table name for embedding rows (default: 'document_embeddings')",
    )
    args = parser.parse_args()

    # Set module-level globals so process_pdf() picks them up
    global _CORPUS, _CHUNK_PREFIX, _TABLE
    _CORPUS = args.corpus if args.corpus else ["public_docs"]
    _CHUNK_PREFIX = args.chunk_prefix if args.chunk_prefix else "public"
    _TABLE = args.table
    log.info("Corpus tags: %s | Chunk prefix: %s | Table: %s", _CORPUS, _CHUNK_PREFIX, _TABLE)

    target = args.path.resolve()

    if target.is_file() and target.suffix.lower() == ".pdf":
        pdf_files = [target]
    elif target.is_dir():
        pdf_files = sorted(target.glob("*.pdf"))
        log.info("Found %d PDFs in %s", len(pdf_files), target)
    else:
        log.error("Not a PDF file or directory: %s", target)
        sys.exit(1)

    if not pdf_files:
        log.error("No PDF files found")
        sys.exit(1)

    results = {"processed": 0, "skipped": 0, "errors": 0, "chunks_total": 0}

    for pdf_path in pdf_files:
        row = process_pdf(
            pdf_path,
            dry_run=args.dry_run,
            force=args.force,
            no_vision=args.no_vision,
            vision_only=args.vision_only,
        )
        if row.get("error"):
            results["errors"] += 1
        elif row.get("_skipped"):
            results["skipped"] += 1
        else:
            results["processed"] += 1
            results["chunks_total"] += int(row.get("chunks_produced", 0))

    log.info("━" * 60)
    log.info(
        "DONE: %d processed, %d skipped, %d errors, %d total chunks",
        results["processed"], results["skipped"],
        results["errors"], results["chunks_total"],
    )


if __name__ == "__main__":
    main()
