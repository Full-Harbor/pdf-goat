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

## Public version note

This repo is the sanitized public version.
It keeps the pipeline logic and removes project-specific defaults so the same method can be adapted to any organization or corpus.

## Relationship to Harbor Commons

[`harbor-commons`](https://github.com/Full-Harbor/harbor-commons) is the public product layer.
`pdf-goat` is one of the ingestion tools underneath that surface.

See [METHOD.md](METHOD.md) for the worked example and public adaptation rules.
