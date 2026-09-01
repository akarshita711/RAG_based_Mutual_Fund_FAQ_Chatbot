"""Phase 3 — Embedding (Gemini via google-genai SDK).

Encodes corpus chunks and user queries with a real Gemini embedding model
(``gemini-embedding-001`` by default, see config.EMBEDDING_MODEL). Embeddings
are produced on demand through the google-genai SDK — there is no pre-computed
on-disk embedding file: Chroma (Phase 4) calls this embedding function when
documents are added and when queries are run, which keeps index and query
vectors in the exact same model/dimension space.

Provides a Chroma adapter (``GeminiEmbeddingFunction``) with two task modes:
  - documents  -> task_type RETRIEVAL_DOCUMENT
  - queries    -> task_type RETRIEVAL_QUERY
This is the recommended RAG pattern for ``gemini-embedding-001``.

Usage (Phase 3 verification):
    python -m src.ingest.embedder "example sentence to embed"
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction  # noqa: E402

from src import config  # noqa: E402
from src.gemini import embed_documents, embed_query  # noqa: E402


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma embedding function backed by the Gemini embeddings API.

    ``queries=False`` (default) tags inputs as corpus documents
    (RETRIEVAL_DOCUMENT); ``queries=True`` tags them as search queries
    (RETRIEVAL_QUERY). Both use the same model + output dimension so the
    vectors live in one space.
    """

    def __init__(self, queries: bool = False) -> None:
        self.queries = queries

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        if not input:
            return []
        texts = list(input)
        if self.queries:
            return [embed_query(t) for t in texts]
        return embed_documents(texts)


def doc_embedding_function() -> GeminiEmbeddingFunction:
    """Embedding function used when adding corpus documents to Chroma."""
    return GeminiEmbeddingFunction(queries=False)


def query_embedding_function() -> GeminiEmbeddingFunction:
    """Embedding function used when embedding search queries."""
    return GeminiEmbeddingFunction(queries=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3 — verify Gemini embeddings (prints dim + snippet)")
    parser.add_argument("text", nargs="?", default="elss lock-in period expense ratio",
                        help="sentence to embed")
    args = parser.parse_args()

    vec = embed_query(args.text)
    print(f"model   : {config.EMBEDDING_MODEL}")
    print(f"dim     : {len(vec)} (config.EMBEDDING_DIM={config.EMBEDDING_DIM})")
    print(f"norm    : {sum(v * v for v in vec) ** 0.5:.4f}")
    print(f"snippet : {[round(v, 4) for v in vec[:6]]} ...")
    print("Phase 3 embedding OK")


if __name__ == "__main__":
    main()