"""Text preprocessing and structure-aware chunking for the Harbor Ingest pipeline.

stdlib-only — no LangChain, no NLTK, no external NLP dependencies.

Two public functions
--------------------
preprocess_text(text, is_ocr=True) -> str
    Clean raw text (especially Internet Archive DjVu OCR) before chunking.
    Key operations: NFKC normalization, OCR de-hyphenation, page-number
    stripping, DjVu artifact removal, whitespace normalisation.

chunk_text(text, chunk_size, overlap) -> list[str]
    Structure-aware recursive splitter. Respects paragraph → sentence →
    word boundaries before falling back to character split. Maintains the
    same ~1500-char / 150-char-overlap semantics as the v1 fixed-stride
    chunker but produces cleaner semantic unit boundaries for the vector
    embedding model.

Chunker version: v2_recursive_1500c_150o  (see scripts/config.py CHUNKER_VERSION)
"""
from __future__ import annotations

import re
import unicodedata
from typing import Sequence


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# DjVu form-feed char inserted at every page boundary
_FORM_FEED = "\x0c"

# Sequence of characters used as typographic section dividers in OCR output
_DIVIDER_RE = re.compile(r"[=\-—_]{8,}")

# Hyphenated line-break: a word fragment at end of line joined to the start
# of the next line. E.g. "evan-\ngelical" → "evangelical"
_DEHYPHEN_RE = re.compile(r"(\w)-\n(\w)")

# Standalone page-number lines produced by DjVu/ABBYY OCR.
# Matches lines that are ONLY 1–3 digits (covers pp. 1–999).
# Intentionally excludes 4-digit numbers to avoid removing year headers (1844, 1869…).
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)

# Running headers: short lines (≤ 60 chars) of ALL-CAPS or small-caps often
# emitted between OCR paragraphs. Heuristic: ≥ 80% uppercase + ≤ 8 words.
_HEADER_RE = re.compile(r"^[ \t]*([A-Z][A-Z ,.\-']{4,59})[ \t]*$", re.MULTILINE)

# Repeating em-dash / hyphen runs used as typographic separators
_DASH_RUN_RE = re.compile(r"[—\-]{8,}")

# Excessive whitespace (3+ blank lines → 2 blank lines)
_EXCESS_NL_RE = re.compile(r"\n{3,}")


def preprocess_text(text: str, is_ocr: bool = True) -> str:
    """Clean *text* before chunking.

    Parameters
    ----------
    text:
        Raw text extracted from PDF (pdftotext) or Internet Archive .djvu.txt.
    is_ocr:
        True  — apply OCR-specific fixes (de-hyphenation, page-number stripping,
                 form-feed removal, dash-run removal). Default.
        False — apply only Unicode normalisation + whitespace collapsing.

    Returns
    -------
    str
        Cleaned text ready for chunk_text().
    """
    # 1. Unicode normalisation — collapses ligatures (ﬁ → fi), smart quotes,
    #    narrow no-break spaces, etc. Safe for all inputs.
    text = unicodedata.normalize("NFKC", text)

    if is_ocr:
        # 2. Form-feed → newline (DjVu page boundary marker)
        text = text.replace(_FORM_FEED, "\n")

        # 3. De-hyphenate line breaks: "evan-\ngelical" → "evangelical"
        text = _DEHYPHEN_RE.sub(r"\1\2", text)

        # 4. Remove standalone page-number lines
        text = _PAGE_NUM_RE.sub("", text)

        # 5. Remove typographic dash/equals separators
        text = _DASH_RUN_RE.sub("", text)
        text = _DIVIDER_RE.sub("", text)

        # 6. Strip all-caps running headers (heuristic).
        #    Only remove lines where the token is ≥ 80% uppercase.
        def _is_caps_header(m: re.Match) -> str:
            line = m.group(1)
            alpha = [c for c in line if c.isalpha()]
            if not alpha:
                return m.group(0)
            upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
            word_count = len(line.split())
            if upper_ratio >= 0.80 and word_count <= 8:
                return ""
            return m.group(0)

        text = _HEADER_RE.sub(_is_caps_header, text)

    # 7. Collapse 3+ blank lines → 2
    text = _EXCESS_NL_RE.sub("\n\n", text)

    # 8. Strip leading/trailing whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Ordered list of split points for the recursive strategy, from coarsest to
# finest.  Each entry is (separator, keep_separator).
_SEPARATORS: list[tuple[str, bool]] = [
    ("\n\n",  False),   # paragraph break
    ("\n",    False),   # line break
    (". ",    True),    # sentence end (keep the period)
    ("? ",    True),
    ("! ",    True),
    ("; ",    True),
    (", ",    True),
    (" ",     False),   # word break
    ("",      False),   # character split (last resort)
]


def _split_on(text: str, sep: str, keep: bool) -> list[str]:
    """Split *text* on *sep*, optionally re-attaching the separator."""
    if not sep:
        # Character-level split — return individual characters
        return list(text)
    parts = text.split(sep)
    if keep:
        # Re-attach separator to end of each part except the last
        return [p + sep for p in parts[:-1]] + [parts[-1]]
    return parts


def _merge_splits(
    splits: list[str],
    chunk_size: int,
    overlap: int,
    separator: str = "",
) -> list[str]:
    """Merge *splits* greedily into chunks of at most *chunk_size* characters.

    Adjacent chunks share *overlap* characters at their boundary.
    *separator* is re-inserted between pieces so paragraph / sentence
    boundaries are preserved in the assembled chunk text.
    """
    sep_len = len(separator)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for split in splits:
        split_len = len(split)
        # Account for the separator that would be added between parts
        extra = sep_len if current_parts else 0

        if current_len + extra + split_len > chunk_size and current_parts:
            # Flush current window as a chunk
            chunk = separator.join(current_parts).strip()
            if chunk:
                chunks.append(chunk)

            # Build the overlap tail by walking backwards until we have
            # approximately *overlap* characters
            tail_parts: list[str] = []
            tail_len = 0
            for part in reversed(current_parts):
                part_len = len(part) + (sep_len if tail_parts else 0)
                if tail_len + part_len > overlap:
                    break
                tail_parts.insert(0, part)
                tail_len += part_len

            current_parts = tail_parts
            current_len = tail_len

        current_parts.append(split)
        current_len += split_len + (sep_len if len(current_parts) > 1 else 0)

    # Flush the last window
    if current_parts:
        chunk = separator.join(current_parts).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: Sequence[tuple[str, bool]],
) -> list[str]:
    """Recursively split *text* at progressively finer boundaries.

    Tries each separator in order.  If a split produces sub-pieces that are
    still larger than *chunk_size*, those pieces are recursively split with
    the next separator.  Falls back to character-level splitting.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Absolute fallback: fixed-stride character split
        return _fixed_stride(text, chunk_size, overlap)

    sep, keep = separators[0]
    remaining = list(separators[1:])

    raw_splits = _split_on(text, sep, keep)

    # Filter empty strings
    splits = [s for s in raw_splits if s.strip()]

    if len(splits) <= 1:
        # The separator didn't split anything — try the next one
        return _recursive_split(text, chunk_size, overlap, remaining)

    # Recursively handle any split piece that is still too large
    good_splits: list[str] = []
    for piece in splits:
        if len(piece) > chunk_size:
            good_splits.extend(_recursive_split(piece, chunk_size, overlap, remaining))
        else:
            good_splits.append(piece)

    # Use the separator as the join glue when re-assembling pieces
    # (keep=True means the sep is already attached to each piece, so join with "")
    join_sep = "" if keep else sep
    return _merge_splits(good_splits, chunk_size, overlap, separator=join_sep)


def _fixed_stride(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-stride character split (same semantics as v1 chunker)."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 150,
) -> list[str]:
    """Split *text* into overlapping chunks of ~*chunk_size* characters.

    Strategy (v2 — recursive, structure-aware):
    1. Try paragraph breaks (\\n\\n) first.
    2. Fall back progressively through: line break → sentence → clause →
       word → character.
    3. Merge resulting pieces into windows up to *chunk_size* with *overlap*
       character carry-over.

    Compared to the v1 fixed-stride chunker:
    - Respects paragraph and sentence boundaries → fewer mid-sentence cuts
    - Better for OCR historical text where sentence length varies widely
    - Same chunk_size / overlap semantics → no schema changes required
    - ~5–15% more chunks for the same corpus (sentences straddle windows less)

    Parameters
    ----------
    text:
        Pre-processed text (pass through preprocess_text() first).
    chunk_size:
        Target maximum characters per chunk (default: 1500).
    overlap:
        Characters of carry-over context between adjacent chunks (default: 150).

    Returns
    -------
    list[str]
        Non-empty chunks, each ≤ chunk_size characters (except when a single
        token exceeds chunk_size, which is theoretically possible for very
        long words in OCR output but rare).
    """
    text = text.strip()
    if not text:
        return []

    # Short texts don't need splitting
    if len(text) <= chunk_size:
        return [text]

    return _recursive_split(text, chunk_size, overlap, _SEPARATORS)
