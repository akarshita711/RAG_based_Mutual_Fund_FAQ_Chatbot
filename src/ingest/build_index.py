"""Phase 4 — Vector Store (ChromaDB).

Builds the searchable vector store from the Phase 2 chunk store using real
Gemini embeddings (via ``src.ingest.embedder.GeminiEmbeddingFunction``).

Persistence:
  - Chroma persists to ``data/vector_db/chroma`` (local PersistentClient).
  - Collection: ``config.CHROMA_COLLECTION`` ("hdfc_mf_faq").
  - Distance metric: cosine (``{"hnsw:space": "cosine"}``).

Every chunk is stored with its metadata (chunk_id, url, title, name,
scheme_type, source_type, fetched_at) so Phase 5 retrieval can return the
citation-bearing chunk.

Phase 3 is exercised implicitly here (Chroma embeds each chunk when we add
it); run ``python -m src.ingest.embedder`` first to sanity-check the API.

Script:  python -m src.ingest.build_index [--force]
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from src.ingest.embedder import doc_embedding_function  # noqa: E402

CHUNK_STORE = os.path.join(config.CHUNKS_DIR, "chunks.jsonl")

META_KEYS = ("chunk_id", "url", "title", "name",
             "scheme_type", "source_type", "fetched_at")

ADD_BATCH_SIZE = 8


def get_collection():
    """Return the Chroma collection (requires chromadb)."""
    import chromadb

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    ef = doc_embedding_function()
    existing = None
    try:
        existing = client.get_collection(config.CHROMA_COLLECTION)
    except Exception:  # noqa: BLE001 - collection does not exist yet
        existing = None
    if existing is not None:
        existing.embedding_function = ef
        return existing
    return client.create_collection(
        config.CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(force: bool = False, verbose: bool = True):
    """Embed the Phase 2 chunks and upsert them into Chroma."""
    config.ensure_dirs()
    if not os.path.exists(CHUNK_STORE):
        print("[index] chunk store missing; run Phase 2 (chunker) first",
              file=sys.stderr)
        sys.exit(1)

    import chromadb

    with open(CHUNK_STORE, encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh if line.strip()]
    if not chunks:
        print(f"[index] no chunks in {CHUNK_STORE}", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    if force:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:  # noqa: BLE001 - nothing to delete
            pass

    collection = get_collection()
    existing = collection.count()
    if existing and not force:
        print(f"[index] Chroma collection '{config.CHROMA_COLLECTION}' already "
              f"has {existing} chunks; use --force to rebuild.")
        return collection

    ids = [c["chunk_id"] for c in chunks]
    documents = [c.get("text", "") for c in chunks]
    metadatas = [{k: (c.get(k, "") if c.get(k, "") is not None else "")
                  for k in META_KEYS} for c in chunks]

    for i in range(0, len(ids), ADD_BATCH_SIZE):
        collection.upsert(
            ids=ids[i:i + ADD_BATCH_SIZE],
            documents=documents[i:i + ADD_BATCH_SIZE],
            metadatas=metadatas[i:i + ADD_BATCH_SIZE],
        )
        if verbose:
            print(f"[index] upserted {min(i + ADD_BATCH_SIZE, len(ids))}/{len(ids)} chunks")

    print(f"[index] Chroma '{config.CHROMA_COLLECTION}' ready: "
          f"{collection.count()} chunks, model={config.EMBEDDING_MODEL}, "
          f"dim={config.EMBEDDING_DIM}, metric=cosine")
    print(f"[index] persisted at {config.CHROMA_DIR}")
    return collection


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 — Vector Store (Chroma)")
    parser.add_argument("--force", action="store_true",
                        help="delete and rebuild the collection")
    args = parser.parse_args()

    collection = build_index(force=args.force)
    print(f"[index] store size: {collection.count()} items")


if __name__ == "__main__":
    main()