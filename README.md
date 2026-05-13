# pdf-goat

Canonical PDF ingestion pipeline for OCR, chunking, embeddings, and provenance.

`pdf-goat` is the single-path PDF processor developed inside the Harbor Commons ecosystem: hash the file, extract text, route image-heavy pages through vision when needed, chunk conservatively, and preserve enough provenance that downstream users can inspect what happened.

## Why it exists

Most small organizations treat PDFs as dead weight.
`pdf-goat` treats them as a source system — but only if the ingestion path is disciplined.

## Pipeline

1. Hash + dedupe
2. Text extraction
3. Page classification
4. Table extraction
5. Vision routing for image-only pages
6. Merge extracted content
7. Preprocess noisy OCR text
8. Structure-aware chunking
9. Optional embeddings
10. Optional Supabase upsert
11. Manifest/provenance write

## Included files

- `pdf_goat.py` — main pipeline
- `chunking.py` — structure-aware text preprocessing and chunking
- `config.py` — generic configuration defaults for the public version

## Quick start

```bash
python pdf_goat.py /path/to/file.pdf --dry-run
python pdf_goat.py /path/to/folder --dry-run
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

Other document types that work well: board minutes, bylaws, annual reports, program evaluations, RFP responses.

## Public version note

This repo is the sanitized public version.
It keeps the pipeline logic and removes project-specific defaults so the same method can be adapted to any organization or corpus.

## Relationship to Harbor Commons

[`harbor-commons`](https://github.com/Full-Harbor/harbor-commons) is the public product layer.
`pdf-goat` is one of the ingestion tools underneath that surface.

See [METHOD.md](METHOD.md) for the worked example and public adaptation rules.
