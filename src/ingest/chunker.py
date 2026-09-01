"""Phase 2 — Chunking.

Splits each cleaned page (from the Phase 1 raw store) into overlapping text
chunks, preserving source metadata (`url`, `title`) on every chunk per
docs/architecture.md Phase 2. Output is a chunk store (JSONL) in data/chunks.

Facts like "expense ratio" should stay within a single retrievable chunk, so we
chunk at a character target with a small overlap. Standard-library only.
"""
from __future__ import annotations

import json
import os
import re
import sys
from hashlib import sha1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config  # noqa: E402


# ---------------------------------------------------------------------------
# Configurable chunking params (tunable in Phase 6 / config.py)
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1200        # target chunk size in characters
DEFAULT_CHUNK_OVERLAP = 150      # overlap between chunks in characters
MIN_CHUNK_CHARS = 80             # drop tiny leftover fragments


class Chunker:
    """Sliding-window chunker with sentence/paragraph-aware boundaries."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 min_chars: int = MIN_CHUNK_CHARS) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chars = min_chars

    def split(self, text: str) -> list[str]:
        """Return a list of text chunks covering ``text``."""
        text = self._normalize(text)
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = start + self.chunk_size
            if end >= n:
                # Last piece of text.
                piece = text[start:].strip()
                if len(piece) >= self.min_chars:
                    chunks.append(piece)
                break
            # Back off to a natural boundary (paragraph > sentence > word).
            cut = self._find_cut(text, start, end)
            piece = text[start:cut].strip()
            if len(piece) >= self.min_chars:
                chunks.append(piece)
            if cut <= start or cut >= n:
                break
            start = cut - self.chunk_overlap
            if start < 0:
                start = 0
        return chunks

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        """Light normalization: collapse blank runs but keep paragraph breaks."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _find_cut(self, text: str, start: int, end: int) -> int:
        """Find the best boundary <= end, prioritizing paragraph then sentence."""
        window = text[start:end]

        # Prefer a paragraph break (double newline).
        para = window.rfind("\n\n")
        if para >= self.chunk_size // 2:
            return start + para

        # Then a single newline.
        nl = window.rfind("\n")
        if nl >= self.chunk_size // 2:
            return start + nl

        # Then a sentence end.
        for sep in (". ", "? ", "! "):
            idx = window.rfind(sep)
            if idx >= self.chunk_size // 3:
                return start + idx + 1  # keep the punctuation with prior chunk

        # Fall back on a word boundary.
        space = window.rfind(" ")
        if space > 0:
            return start + space

        return end


def _chunk_id(rec: dict, index: int) -> str:
    material = f"{rec['url']}::{index}"
    return sha1(material.encode("utf-8")).hexdigest()[:16]


def chunk_records(records: list[dict],
                  chunk_size: int = DEFAULT_CHUNK_SIZE,
                  chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                  force: bool = False) -> list[dict]:
    """Chunk raw records from the Phase 1 store.

    Each returned chunk dict carries source metadata plus its text.
    PII is intentionally not copied (per FR-5); we only propagate fields
    needed for citation (url, name, scheme_type, source_type, fetched_at).
    """
    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[dict] = []
    for rec in records:
        text = rec.get("text") or ""
        parts = chunker.split(text)
        base = {
            "name": rec.get("name", ""),
            "scheme_type": rec.get("scheme_type", ""),
            "source_type": rec.get("source_type", "scheme"),
            "title": rec.get("title", ""),
            "url": rec.get("url", ""),
            "fetched_at": rec.get("fetched_at", ""),
        }
        for i, part in enumerate(parts):
            chunk = dict(base)
            chunk["chunk_id"] = _chunk_id(rec, i)
            chunk["chunk_index"] = i
            chunk["text"] = part
            chunk["char_count"] = len(part)
            chunk["word_count"] = len(part.split())
            chunks.append(chunk)
    return chunks


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 — Chunking")
    parser.add_argument("--force", action="store_true",
                        help="regenerate chunks even if the store exists")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    config.ensure_dirs()
    if not os.path.exists(config.RAW_STORE):
        print("[chunker] raw store missing; run Phase 1 first", file=sys.stderr)
        sys.exit(1)

    with open(config.RAW_STORE, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    out_path = os.path.join(config.CHUNKS_DIR, "chunks.jsonl")
    if os.path.exists(out_path) and not args.force:
        print(f"[chunker] chunks exist ({out_path}); use --force to rebuild")
        return

    chunks = chunk_records(records,
                           chunk_size=args.chunk_size,
                           chunk_overlap=args.chunk_overlap)
    with open(out_path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_scheme = {}
    for c in chunks:
        by_scheme.setdefault(c["scheme_type"], 0)
        by_scheme[c["scheme_type"]] += 1
    print(f"[chunker] wrote {len(chunks)} chunks -> {out_path}")
    for k, v in by_scheme.items():
        print(f"  {k}: {v} chunks")


if __name__ == "__main__":
    main()
