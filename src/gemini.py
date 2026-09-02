"""Google Gemini SDK integration (google-genai).

Single home for the Gemini client used by:
  - Phase 3 embedding  (embed_documents / embed_query)
  - Phase 4 vector store build (via embedder.GeminiEmbeddingFunction)
  - Phase 7 generation  (generate_text)

Everything reads credentials/settings from config.py (which loads .env).
"""
from __future__ import annotations

import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from google import genai  # noqa: E402
from google.genai import errors, types  # noqa: E402

from src import config  # noqa: E402

# Pacing for the (often low) free-tier embedding quotas. Keep the batch small
# and pause between batches so we stay under per-minute request limits on a
# base-model basis; retry 429s with exponential backoff.
EMBED_BATCH_SIZE = int(os.environ.get("GEMINI_EMBED_BATCH_SIZE", "8"))
EMBED_BATCH_SLEEP = float(os.environ.get("GEMINI_EMBED_BATCH_SLEEP", "3"))
_429_BACKOFF = [5, 10, 20, 40, 80]  # seconds between retries on RESOURCE_EXHAUSTED
_GEN_BACKOFF = [3, 6, 10]  # shorter retry ladder for interactive generation


class LLMError(RuntimeError):
    """Raised for any Gemini SDK/API failure (auth, network, model, response)."""


_client = None
_client_missing = False


def get_client():
    """Return a cached genai.Client, or None when no API key is configured."""
    global _client, _client_missing
    if _client is not None:
        return _client
    if _client_missing:
        return None

    key = os.environ.get(config.GEMINI_API_KEY_ENV, "").strip()
    if not key:
        _client_missing = True
        return None

    http_options = None
    base = os.environ.get(config.GEMINI_BASE_URL_ENV, "").strip().strip("/")
    if base:
        # Accept a base URL like https://host/v1beta -> split api version out.
        api_version = None
        match = re.search(r"/(v1(?:beta)?)$", base)
        if match:
            api_version = match.group(1)
            base = base[: match.start()].rstrip("/")
        if base:
            http_options = types.HttpOptions(
                api_version=api_version or "v1",
                base_url=base,
            )
    try:
        _client = (genai.Client(api_key=key, http_options=http_options)
                   if http_options else genai.Client(api_key=key))
    except Exception as exc:  # noqa: BLE001 - normalize SDK init failures
        raise LLMError(f"Gemini client init failed: {exc}") from exc
    return _client


# ---------------------------------------------------------------------------
# Embedding (Phase 3)
# ---------------------------------------------------------------------------
def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    client = get_client()
    if client is None:
        raise LLMError("Gemini embeddings require GOOGLE_API_KEY to be set.")
    if not texts:
        return []

    attempt = 0
    while True:
        try:
            result = client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=list(texts),
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=config.EMBEDDING_DIM,
                ),
            )
            break
        except errors.ClientError as exc:
            if exc.code == 429 and attempt < len(_429_BACKOFF):
                wait = _429_BACKOFF[attempt]
                print(f"[gemini] embedding rate-limited (429); backoff {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
                attempt += 1
                continue
            raise LLMError(f"Gemini embed_content failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Gemini embed_content failed: {exc}") from exc

    embeddings = []
    for item in result.embeddings:
        values = getattr(item, "values", None)
        if values is None:
            raise LLMError("Gemini returned embeddings without values.")
        embeddings.append([float(v) for v in values])
    return embeddings


def embed_documents(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Embed corpus chunks (task_type=RETRIEVAL_DOCUMENT), paged + paced."""
    batch_size = batch_size or EMBED_BATCH_SIZE
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(_embed_batch(texts[i:i + batch_size], "RETRIEVAL_DOCUMENT"))
        if i + batch_size < len(texts):
            time.sleep(EMBED_BATCH_SLEEP)
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a user query (task_type=RETRIEVAL_QUERY)."""
    result = _embed_batch([text], "RETRIEVAL_QUERY")
    return result[0] if result else []


# ---------------------------------------------------------------------------
# Generation (Phase 7)
# ---------------------------------------------------------------------------
def generate_text(system: str, user: str) -> str:
    """Generate a completion with system_instruction + user content."""
    client = get_client()
    if client is None:
        raise LLMError("No Gemini API key configured (GOOGLE_API_KEY).")
    attempt = 0
    while True:
        try:
            response = client.models.generate_content(
                model=config.LLM_MODEL,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=config.LLM_TEMPERATURE,
                    max_output_tokens=1024,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True,
                    ),
                ),
            )
            break
        except errors.APIError as exc:
            # Transient 429 (rate limit) / 503 (high demand): back off and retry.
            code = getattr(exc, "code", None)
            if code in (429, 503) and attempt < len(_GEN_BACKOFF):
                wait = _GEN_BACKOFF[attempt]
                print(f"[gemini] generate_content {code}; backoff {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
                attempt += 1
                continue
            raise LLMError(f"Gemini generate_content failed: "
                           f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - normalize SDK failures
            raise LLMError(f"Gemini generate_content failed: "
                           f"{type(exc).__name__}: {exc}") from exc
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise LLMError("Gemini returned an empty response.")
    return text


__all__ = [
    "LLMError",
    "get_client",
    "embed_documents",
    "embed_query",
    "generate_text",
    "errors",
    "types",
]