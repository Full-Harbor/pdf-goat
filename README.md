# pdf-goat

Canonical PDF ingestion pipeline for OCR, chunking, embeddings, and provenance.

**Current version: v2.0.3**

`pdf-goat` is the single-path PDF processor developed inside the Harbor Commons
ecosystem: hash the file, extract text, route image-heavy pages through vision
when needed, chunk conservatively, and preserve enough provenance that downstream
users can inspect what happened.

## Why it exists

Most small organizations treat PDFs as dead weight.
`pdf-goat` treats them as a source system — but only if the ingestion path is
disciplined.

## Pipeline (12 steps)

1. **Hash + dedupe** — SHA256; skip if already in manifest
2. **Text extraction** — Docling primary, pdfplumber fallback
3. **Page classification** — `text_only` / `text_with_images` / `image_only` / `empty`
4. **Table extraction** — Docling TableFormer primary, pdfplumber fallback
5. **Vision routing** — `image_only` → full_vision; mixed → page_vision
6. **Claude Vision extraction** — for flagged pages only
7. **Merge** — text + vision output per page
8. **Preprocess** — `chunking.preprocess_text`
9. **Chunk** — `chunking.chunk_text` (v2 recursive, ~1,500 char / 150 overlap)
10. **Embed** — `text-embedding-3-large` @ 1536 dimensions
11. **Upsert** — Supabase `sailing_embeddings` table, on-conflict dedupe
12. **Manifest** — local CSV provenance row per file

## Content type schema

Each extracted chunk carries a `content_type` field:

| Value | Meaning |
|-------|---------|
| `text_only` | Clean text, no tables or images |
| `table` | Tabular data only |
| `text_with_tables` | Mixed prose + tables |
| `vision_table` | Table reconstructed from Claude Vision (image-only page) |

## Corpus field

Each chunk also carries a `corpus` field (`text[]`) for downstream filtering —
e.g. `["sailing", "annual_report", "oia"]`.

## Included files

- `pdf_goat.py` — main 12-step pipeline
- `chunking.py` — structure-aware preprocessing and recursive chunking
- `config.py` — configuration defaults (sanitized for public use)
- `requirements.txt` — Python dependencies

## Quick start

```bash
# Single file
python pdf_goat.py /path/to/doc.pdf

# Directory (recursive)
python pdf_goat.py /path/to/folder/

# Dry run — extract + manifest only, no embeddings or Supabase
python pdf_goat.py /path/to/doc.pdf --dry-run

# Skip vision (text-only, no API calls)
python pdf_goat.py /path/to/doc.pdf --no-vision

# Force reprocess (ignore manifest/dedupe)
python pdf_goat.py /path/to/doc.pdf --force

# Vision-only (only process image_only pages)
python pdf_goat.py /path/to/doc.pdf --vision-only
```

## Worked example

A community foundation receives 40 grant reports as scanned PDFs.
Some are clean text; some are image-only scans of faxed forms.

```bash
# Dry run to classify pages and preview chunk counts
python pdf_goat.py ./grant_reports/ --dry-run

# Full run: extract, chunk, embed, upsert to Supabase
python pdf_goat.py ./grant_reports/ \
  --table document_embeddings \
  --batch-label "grant_reports_fy24"
```

The pipeline will:
- Hash each file and skip duplicates
- Route image-only pages through Claude Vision
- Chunk each document at ~1,500 characters with 150-character overlap
- Write a provenance manifest so you know exactly which pages were vision-processed vs. text-extracted

Other document types that work well: board minutes, bylaws, annual reports,
program evaluations, RFP responses.

## Backend selection

| Backend | Primary use | Fallback |
|---------|-------------|---------|
| Docling | Text extraction + TableFormer tables | — |
| pdfplumber | Text extraction fallback | ✅ auto |
| pdfplumber | Table extraction fallback | ✅ auto |
| Claude Vision | Image-only and mixed pages | — |

Docling is preferred when available. pdfplumber is the graceful fallback for
environments where Docling is not installed or fails on a specific file.

## Public version note

This repo is the sanitized public version.
It keeps the pipeline logic and removes project-specific defaults so the same
method can be adapted to any organization or corpus.

## Relationship to Harbor Commons

[`harbor-commons`](https://github.com/Full-Harbor/harbor-commons) is the public
product layer. `pdf-goat` is one of the ingestion tools underneath that surface.

See [METHOD.md](METHOD.md) for the worked example and public adaptation rules.

## Changelog

### v2.0.3
- Docling/pdfplumber dual backend with automatic fallback
- `content_type` schema: `text_only`, `table`, `text_with_tables`, `vision_table`
- `corpus` field (`text[]`) on all chunks for downstream filtering
- Recursive chunking v2 in `chunking.py`
- 12-step pipeline (was 11)
- `config.py` expanded with embedding model, chunking, and vision parameters
