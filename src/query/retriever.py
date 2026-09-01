"""Phase 5 — Retrieval Logic (ChromaDB + Gemini query embeddings).

Encodes the user query with the Gemini query-embedding function and searches
the Phase 4 Chroma collection (cosine space). Applies a minimum-similarity
threshold -> the "no-source" refusal path.

Per docs/architecture.md Phase 5:
  - Encode the user query into a vector.
  - Retrieve top-k most similar chunks.
  - If no chunk scores above a minimum threshold -> no-source refusal path.

Chroma returns cosine *distance*; similarity = 1 - distance.
"""
from __future__ import annotations

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from src.ingest.embedder import query_embedding_function  # noqa: E402


class Retriever:
    """Retrieves candidate chunks for a user query from the Chroma store."""

    def __init__(self, top_k: int | None = None,
                 min_score: float | None = None) -> None:
        self.top_k = top_k if top_k is not None else config.RETRIEVER_TOP_K
        self.min_score = (min_score if min_score is not None
                          else config.RETRIEVER_MIN_SCORE)
        self.collection = None
        self.init_error = None
        try:
            self.collection = self._load_collection()
        except Exception as exc:  # noqa: BLE001 - surface as retrieval failure
            self.init_error = exc

    # -- setup ---------------------------------------------------------------
    def _load_collection(self):
        import chromadb

        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        ef = query_embedding_function()
        try:
            collection = client.get_or_create_collection(
                config.CHROMA_COLLECTION,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            return collection
        except Exception:  # noqa: BLE001 - existing collection w/ diff config
            collection = client.get_collection(config.CHROMA_COLLECTION)
            collection.embedding_function = ef
            return collection

    # -- retrieval -----------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None,
                 min_score: float | None = None) -> dict:
        """Rank chunks for ``query``.

        Returns a dict:
          {
            "query": str,
            "hits": [ {chunk metadata..., "text", "score", "distance"}, ... ],
            "scores": {"top": float, "min": float, "lowest": float},
            "refused": bool,          # True when nothing / below threshold
            "reason": str,            # explains refusal / success
          }
        """
        k = k if k is not None else self.top_k
        min_score = min_score if min_score is not None else self.min_score

        if self.init_error is not None or self.collection is None:
            return self._refusal(query, "vector_store_unavailable")

        results = self._query_chroma(query, k)
        if results == "embedding":
            return self._refusal(query, "embedding_unavailable")
        if results == "query":
            return self._refusal(query, "query_failed")

        hits = self._build_hits(results)
        refused = False
        reason = "retrieved"
        if not hits:
            refused = True
            reason = "no_chunks"
        else:
            top = hits[0]["score"]
            if top < min_score:
                refused = True
                reason = "below_threshold"

        return {
            "query": query,
            "hits": hits,
            "scores": {
                "top": round(hits[0]["score"], 4) if hits else 0.0,
                "min": min_score,
                "lowest": round(hits[-1]["score"], 4) if hits else 0.0,
            },
            "refused": refused,
            "reason": reason,
        }

    def _query_chroma(self, query: str, k: int) -> dict | str:
        """Query Chroma.

        Returns the raw response dict on success, or a failure marker string:
        ``"embedding"`` (Gemini embed error) or ``"query"`` (Chroma error).
        """
        from src.gemini import LLMError

        try:
            return self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except LLMError:
            return "embedding"
        except Exception as exc:  # noqa: BLE001
            print(f"[retriever] chroma query failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return "query"

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _first(rows) -> list:
        """Chroma may return nested lists (one row per query); flatten to one."""
        if rows and isinstance(rows[0], list):
            return rows[0]
        return rows or []

    def _build_hits(self, results: dict) -> list[dict]:
        ids = self._first(results.get("ids"))
        distances = self._first(results.get("distances"))
        metadatas = self._first(results.get("metadatas"))
        documents = self._first(results.get("documents"))

        hits = []
        for i, cid in enumerate(ids):
            meta = dict(metadatas[i] or {}) if i < len(metadatas) else {}
            dist = float(distances[i]) if i < len(distances) else 0.0
            text = documents[i] if i < len(documents) else ""
            hit = {**meta, "chunk_id": cid, "text": text,
                   "score": round(1.0 - dist, 4), "distance": dist}
            hits.append(hit)
        return hits

    @staticmethod
    def _refusal(query: str, reason: str) -> dict:
        return {
            "query": query,
            "hits": [],
            "scores": {"top": 0.0, "min": 0.0, "lowest": 0.0},
            "refused": True,
            "reason": reason,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5 — Retrieval Logic")
    parser.add_argument("query", help="user query to retrieve for")
    parser.add_argument("--k", type=int, default=None, help="override top-k")
    parser.add_argument("--min-score", type=float, default=None,
                        help="override minimum threshold")
    args = parser.parse_args()

    retriever = Retriever(top_k=args.k, min_score=args.min_score)
    out = retriever.retrieve(args.query)

    print(f"\nQUERY: {out['query']}")
    print(f"refused={out['refused']} reason={out['reason']} "
          f"scores={out['scores']}")
    for rank, hit in enumerate(out["hits"], start=1):
        snippet = hit["text"][:90].replace("\n", " ")
        print(f"  #{rank} score={hit['score']:.4f} "
              f"[{hit.get('scheme_type','')}] {snippet}")
        print(f"       url={hit['url']}")
    if not out["hits"]:
        print("  (no candidate chunks - no-source refusal path)")


if __name__ == "__main__":
    main()