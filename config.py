"""Generic configuration defaults for the public pdf-goat pipeline.

This public version keeps the operational knobs that matter for chunking,
embeddings, and retry behavior, while removing project-specific corpus and
research notes from the private working repo.
"""

EMBEDDING_MODEL: str = "text-embedding-3-large"
EMBED_DIMENSIONS: int = 1536

CHUNK_CHARS: int = 1500
CHUNK_OVERLAP: int = 150
CHUNKER_VERSION: str = "v2_recursive_1500c_150o"

IMPORT_BATCH_VERSION: str = "pdf_goat_public_v1"

MAX_RETRIES: int = 5
RETRY_BASE_WAIT: float = 1.0
