"""Shared configuration constants for Harbor Ingest embedding pipeline.

Single source of truth for all embedding, chunking, retrieval, and retry
parameters used across ingest_ymca_docs.py, ussailing_embed.py, and ask.py.

--- Paper registry (all verified Mar 30, 2026 via arXiv direct fetch) ---

MRL — Kusupati et al., NeurIPS 2022, arXiv:2205.13147
  Trains a single encoder so every prefix of the embedding is independently
  useful. The multi-scale contrastive loss forces high-signal information
  into early dimensions, enabling truncation without full re-training.

SMEC — Zhang et al., EMNLP 2025, arXiv:2510.12474
  "SMEC: Rethinking Matryoshka Representation Learning for Retrieval
  Embedding Compression." Sequential compression replaces parallel
  multi-scale training. Three new modules: SMRL (reduces gradient
  variance), ADS (minimises information loss during pruning), S-XBM
  (improves unsupervised high-/low-dim alignment). On BEIR at 256d,
  improves LLM2Vec by +1.1 NDCG@10 over Matryoshka-Adaptor and +2.7
  over Search-Adaptor.
  ACTIONABILITY: NOT applicable to OpenAI API usage. Requires model
  fine-tuning + labeled retrieval data + GPU.
  RECEIPT — SMEC Appendix B, Table 4 (BEIR sub-datasets, NDCG@10,
  LLM2Vec base model; URL: arxiv.org/html/2510.12474v1#A2):
    Scifact  @ 1536d: SMEC=0.885  MRL-Adaptor=0.886  (diff: −0.001)
    FiQA     @ 1536d: SMEC=0.549  MRL-Adaptor=0.547  (diff: +0.002)
    Quora    @ 1536d: SMEC=0.862  MRL-Adaptor=0.862  (diff:  0.000)
    NFCorpus @ 1536d: SMEC=0.430  MRL-Adaptor=0.426  (diff: +0.004)
    SciDocs  @ 1536d: SMEC=0.261  MRL-Adaptor=0.262  (diff: −0.001)
  At 128d the same table shows SMEC vs MRL-Adaptor gaps of 0.015–0.057
  across those same sub-datasets — gains are concentrated at low dims.
  Note: these rows test SMEC applied to LLM2Vec, not text-embedding-3-
  large directly. Figure 5 also shows OpenAI curves but without a
  per-sub-dataset table. Treat the 1536d convergence as directionally
  valid for our use case, not as a precise numeric guarantee.
  Relevant only when Full Harbor trains an open-weight encoder.

TMRL — Huynh et al., arXiv Jan 9 2026, arXiv:2601.05549
  "Efficient Temporal-aware Matryoshka Adaptation for Temporal
  Information Retrieval." Injects a temporal subspace into the Matryoshka
  nested structure (first t=64-128 dims encode temporal signals); uses
  LoRA + temporal contrastive loss + self-distillation. Temporal gains
  are strongest at low dims (64-128d); at full 768d the gap vs vanilla
  LoRA-MRL shrinks. Tested on 6 open-weight TEMs (contriever, BGE, etc.).
  ACTIONABILITY: NOT applicable to OpenAI API usage. Requires LoRA
  fine-tuning of an open-weight TEM + temporal training data (e.g.
  augmented TNP dataset). At 1536d the base model likely encodes adequate
  temporal signal already. Relevant when Full Harbor goes self-hosted
  and the YMCA corpus (1844-2026) needs date-range queries to work well.

Matryoshka SAEs — Bussmann, Nabeshima, Karvonen, Nanda; arXiv:2503.17547
  "Learning Multi-Level Features with Matryoshka Sparse Autoencoders."
  Trains nested SAE dictionaries so smaller ones capture broad concepts,
  larger ones capture specifics — reduces feature absorption/splitting.
  Demonstrated on Gemma-2-2B. Venue: unverified — arXiv HTML header
  reads "Machine Learning, ICML" but that string may be a LaTeXML
  renderer template, not a confirmed acceptance. Canon requires an
  OpenReview link or PMLR proceedings URL. Treat as unconfirmed until
  then.
  ACTIONABILITY: NOT applicable to the retrieval pipeline. SAEs are a
  mechanistic interpretability tool for LLM *internal activations*, not
  for embedding-based retrieval. Would require self-hosted model +
  separate SAE training pass to access activations. Far-future research
  direction only.

--- 89.5% benchmark — sourced, verbatim quote from Supabase blog ---
  URL: supabase.com/blog/matryoshka-embeddings
  Section: "1536 Dimensional Vectors without Second Pass"
  Quote: "We obtained an accuracy of 89.5% with KNN search, signifying
  the maximum possible accuracy for vectors shortened to 1536
  dimensions."
  Dataset: 1M text-embedding-3-large embeddings of dbpedia texts.
  Accuracy definition: fraction of top-10 KNN IDs at 1536d matching
  top-10 KNN at full 3072d on *that dataset*.
  CAVEAT: this is neighbor-agreement on Supabase's benchmark, not
  retrieval quality on our tasks. It supports "1536 is operationally
  sane," not "1536 is always best." Do not cite it as retrieval quality.

--- Trained granularities for text-embedding-3-large ---
  OpenAI has not publicly stated exact trained breakpoints.
  Supabase infers from MTEB performance curves that 256, 1024, and 3072
  are likely trained granularities — this is Supabase's inference, not
  an OpenAI claim. 1536 is NOT on that inferred list; Supabase calls it
  "a likely sub-vector granularity" and tests it because pgvector
  indexes max out at 2000d. The 89.5% neighbor-agreement result shows
  1536 is operationally viable, not that it is a trained breakpoint.
  Our vector(1536) schema and zero-migration policy stay intact.

--- Standing policy: no quote, no claim ---
  Every nontrivial factual claim in this file requires:
    - verbatim quote
    - source URL
    - section or table reference in the source document
  Absent those, the claim must be softened to inference or removed.
  Unverifiable claims that survive review must carry an explicit
  "INFERENCE / NOT SOURCED" tag so future readers know the status.

--- Non-negotiable rule (operational) ---
  Query embeddings MUST match corpus embeddings: same model, same
  dimensions= value, same API call pattern. Never manually truncate
  stored vectors. The dimensions= parameter is the correct mechanism;
  do not assert additional normalisation behavior without an OpenAI
  source citation.

--- Pending: evaluation harness ---
  The three papers above are most useful as a *specification for what
  to measure*, even where their methods are not directly actionable:
    - hit-rate@k on a gold query set (ground truth doc IDs known)
    - citation fidelity: does the top-k result contain the answer span?
    - compression failure modes: which query types degrade at 1536d?
  TODO: build gold query set of 20-40 YMCA queries with known answers
  before any future model or dim change. No evaluation → no comparison.
"""

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str    = "text-embedding-3-large"
EMBED_DIMENSIONS: int   = 1536   # use API dimensions= param; never manually truncate stored vectors
COST_PER_1M_TOK: float  = 0.13   # USD — text-embedding-3-large, 2025 pricing

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_CHARS: int     = 1500   # target characters per chunk
CHUNK_OVERLAP: int   = 150    # overlap between adjacent chunks

# Human-readable version stamp written to DB rows.
# Bump when chunk strategy changes (triggers delta-aware re-embed).
CHUNKER_VERSION: str = "v2_recursive_1500c_150o"

# ---------------------------------------------------------------------------
# Corpus versioning stamps (written to embedding_model / import_batch fields)
# Update month part when re-embedding a whole corpus from scratch.
# ---------------------------------------------------------------------------
YMCA_EMBED_VERSION: str    = "ymca_v2_large1536_2026-03"
SAILING_EMBED_VERSION: str = "sailing_v2_large1536_2026-03"

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------
DEFAULT_LIMIT: int      = 8
DEFAULT_THRESHOLD: float = 0.3

# YMCA working-set shelves
# Default experience should prefer the active analysis shelf (curated lineage,
# histories, proceedings, yearbooks, and other high-signal research docs).
# Deep archive remains queryable on demand; nothing is deleted.
YMCA_DEFAULT_SCOPE: str = "active_analysis"
YMCA_ACTIVE_ANALYSIS_DOC_TYPES: tuple[str, ...] = (
  "research_synthesis",
  "historical_proceedings",
  "institutional_history",
  "contemporary_periodical",
  "annual_yearbook",
  "budget",
  "metrics",
  "annual_report",
  "impact_report",
  "strategy_report",
  "world_service",
)
YMCA_DEEP_ARCHIVE_DOC_TYPES: tuple[str, ...] = (
  "historical_annual_report",
  "historical_minutes",
  "historical_journal",
  "historical_document",
)

# BM25 re-ranking hyperparameters
# Retrieve this many vector candidates, then rerank down to DEFAULT_LIMIT.
RERANK_FETCH_K: int   = 60    # wider net for sparse+dense fusion
RERANK_ALPHA: float   = 0.55  # weight for cosine sim; (1-alpha) for BM25

# BM25 Robertson-Spärck-Jones parameters
BM25_K1: float = 1.5   # term-frequency saturation
BM25_B: float  = 0.75  # document-length normalisation

# ---------------------------------------------------------------------------
# API retry
# ---------------------------------------------------------------------------
MAX_RETRIES: int       = 5
RETRY_BASE_WAIT: float = 1.0   # seconds; doubles each attempt (exponential)
