# pdf-goat — Method

## Purpose

`pdf-goat` exists to answer a practical question:
How do you turn messy PDFs into usable, reviewable infrastructure without pretending the extraction was cleaner than it was?

## Core principles

### 1. One canonical path
If a team has five slightly different PDF scripts, it has no PDF pipeline.
A single canonical path reduces hidden drift.

### 2. Provenance over vibes
Every processed file should preserve enough metadata that a later reviewer can understand what was extracted, what required OCR or vision, and whether the result is trustworthy.

### 3. Structure-aware chunking
Chunking is not just a token-count exercise.
The public version keeps paragraph/sentence-aware chunking because downstream retrieval quality depends on preserving document structure.

### 4. Optional infrastructure hooks
Embeddings and database writes are useful, but they should be optional.
The public version is designed so teams can run dry, inspect outputs, and then wire in their own storage.

## Worked example

Two generic documents the method handles well:
- a nonprofit Form 990 PDF with mixed text and tables
- a board minutes packet that includes OCR noise and scanned pages

In both cases, the sequence is the same:
1. hash the PDF
2. extract page-level text
3. route image-only pages through vision if enabled
4. normalize OCR noise
5. chunk into inspectable units
6. optionally embed or upsert to your own store

## Public adaptation notes

The original working version carried corpus-specific defaults for table names, tags, and chunk prefixes.
The public version strips those assumptions so the method can travel.

That is the broader point of the repo:
not "PDFs for sailing," but a reusable PDF discipline for any organization that lacks dedicated data engineering support.
