"""Phase 1 — Data Loading.

Fetches official public pages, extracts readable text (with a best-effort
HTML-to-text conversion), and persists a raw document store (JSONL) plus a
human-readable source list (CSV/MD).

Scope: docs/architecture.md Phase 1. Standard-library only (no third-party deps).
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config  # noqa: E402


# ---------------------------------------------------------------------------
# HTML -> text extraction (stdlib HTMLParser, Dropbox-style block handling)
# ---------------------------------------------------------------------------
class _TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
        "tr", "td", "th", "section", "article", "table",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "iframe"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of whitespace/newlines into single newlines.
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n", raw)
        return raw.strip()


def html_to_text(markup: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(markup or "")
    extractor.close()
    return extractor.text()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def _safe_filename(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9]+", "-", url.rstrip("/").split("/")[-1]).strip("-")
    stem = (stem or "page")[:60]
    return f"{stem}-{digest}.html"


def fetch_page(url: str, cache_dir: str | None = None) -> bytes:
    """Fetch a URL, optionally falling back to a saved HTML cache file.

    Returns the raw HTML bytes. When ``cache_dir`` is given, a local copy is
    saved on success and reused on later runs (crash-resilient / offline).
    """
    if cache_dir:
        cache_path = os.path.join(cache_dir, _safe_filename(url))
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fh:
                return fh.read()

    request = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    with urllib.request.urlopen(request, timeout=config.HTTP_TIMEOUT_SECONDS) as resp:
        raw = resp.read()

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, _safe_filename(url)), "wb") as fh:
            fh.write(raw)
    return raw


# ---------------------------------------------------------------------------
# Source list generation
# ---------------------------------------------------------------------------
def write_source_csv(rows, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["name", "scheme_type", "source_type", "url", "status", "fetched_at"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_source_md(rows, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Source List — HDFC Mutual Fund FAQ RAG\n\n")
        fh.write("Public sources only (per PRD NFR-2).\n\n")
        fh.write("| Name | Scheme Type | Source Type | URL | Status | Fetched At |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(
                f"| {r['name']} | {r['scheme_type']} | {r['source_type']} "
                f"| {r['url']} | {r['status']} | {r['fetched_at']} |\n"
            )


# ---------------------------------------------------------------------------
# Main load routine
# ---------------------------------------------------------------------------
def load_all(urls=None, cache_dir: str | None = None, force: bool = False) -> list[dict]:
    """Fetch and persist the corpus.

    - ``urls``: list of dicts with keys ``name``, ``scheme_type``, ``source_type``, ``url``.
      Defaults to config.SCHEMES.
    - ``cache_dir``: optional dir to save raw HTML copies.
    - ``force``: refetch even if already in the raw store.
    """
    config.ensure_dirs()
    if urls is None:
        urls = config.SCHEMES
    if cache_dir is None:
        cache_dir = os.path.join(config.RAW_DIR, "html_cache")

    existing = {}
    if not force and os.path.exists(config.RAW_STORE):
        with open(config.RAW_STORE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing[rec["url"]] = rec

    records = []
    source_rows = []
    for item in urls:
        url = item["url"]
        row = {
            "name": item["name"],
            "scheme_type": item.get("scheme_type", ""),
            "source_type": item.get("source_type", "scheme"),
            "url": url,
            "status": "error",
            "fetched_at": "",
        }
        try:
            if url in existing:
                rec = existing[url]
                rec["_cache"] = True
            else:
                raw_html = fetch_page(url, cache_dir=cache_dir)
                text = html_to_text(raw_html.decode("utf-8", errors="replace"))
                title = _guess_title(raw_html, text)
                rec = {
                    "url": url,
                    "title": title,
                    "name": item["name"],
                    "scheme_type": item.get("scheme_type", ""),
                    "source_type": item.get("source_type", "scheme"),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                    "word_count": len(text.split()),
                }
            row["status"] = "ok" if rec else "error"
            row["fetched_at"] = rec.get("fetched_at", "")
            records.append(rec)
        except urllib.error.HTTPError as exc:
            row["status"] = f"http-{exc.code}"
            print(f"[loader] HTTP {exc.code} for {url}", file=sys.stderr)
        except urllib.error.URLError as exc:
            row["status"] = "net-error"
            print(f"[loader] network error for {url}: {exc.reason}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - keep ingestion resilient
            row["status"] = "error"
            print(f"[loader] error for {url}: {exc}", file=sys.stderr)
        source_rows.append(row)

    with open(config.RAW_STORE, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_source_csv(source_rows, config.SOURCE_LIST_CSV)
    write_source_md(source_rows, config.SOURCE_LIST_MD)

    ok = sum(1 for r in source_rows if r["status"] == "ok")
    print(f"[loader] loaded {ok}/{len(source_rows)} pages -> {config.RAW_STORE}")
    return records


def _guess_title(raw_html: bytes, text: str) -> str:
    """Best-effort <title> extraction via regex on a small sample."""
    sample = raw_html[:65536].decode("utf-8", errors="replace")
    m = re.search(r"<title[^>]*>(.*?)</title>", sample, re.IGNORECASE | re.DOTALL)
    if m:
        title = html.unescape(m.group(1)).strip()
        if title:
            return title
    return text.split("\n", 1)[0].strip()[:120] if text else ""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1 — Data Loading")
    parser.add_argument("--force", action="store_true", help="refetch pages already in the store")
    parser.add_argument("--no-cache", action="store_true", help="do not persist raw HTML cache")
    args = parser.parse_args()

    load_all(force=args.force, cache_dir=None if args.no_cache else None)


if __name__ == "__main__":
    main()
