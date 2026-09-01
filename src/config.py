"""Central configuration for the HDFC Mutual Fund FAQ RAG prototype.

Scope is restricted to docs/PRD.md / docs/architecture.md.
"""
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# ---------------------------------------------------------------------------
# .env loader (stdlib, no python-dotenv dependency)
# Loads KEY=VALUE pairs from <project root>/.env into the process environment
# WITHOUT overwriting variables already set in the shell.
# ---------------------------------------------------------------------------
_DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(_DOTENV_PATH):
    _DOTENV_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")
    with open(_DOTENV_PATH, encoding="utf-8") as _fh:
        for _line in _fh:
            _m = _DOTENV_RE.match(_line)
            if _m and not _line.lstrip().startswith("#"):
                _key, _val = _m.group(1), _m.group(2).strip().strip("\"'")
                if _key not in os.environ and _val:
                    os.environ[_key] = _val

# ---------------------------------------------------------------------------
# Directory layout (per docs/architecture.md)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")
CHROMA_DIR = os.path.join(VECTOR_DB_DIR, "chroma")
CHROMA_COLLECTION = "hdfc_mf_faq"

# Legacy artifacts from the pre-chroma implementation (no longer written).
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

RAW_STORE = os.path.join(RAW_DIR, "documents.jsonl")

SOURCE_LIST_CSV = os.path.join(RAW_DIR, "sources.csv")
SOURCE_LIST_MD = os.path.join(RAW_DIR, "sources.md")

# ---------------------------------------------------------------------------
# Corpus seed (PRD section 3) — canonical scheme -> official URL mapping
# ---------------------------------------------------------------------------
# source_type labels follow architecture Phase 1 metadata expectations.
SCHEMES = [
    {
        "name": "HDFC Large Cap Fund - Direct Growth",
        "scheme_type": "large-cap",
        "source_type": "scheme",
        "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    },
    {
        "name": "HDFC Flexi Cap Fund - Direct Growth",
        "scheme_type": "flexi-cap",
        "source_type": "scheme",
        # Note: Groww still uses the legacy "equity fund" slug from the old name.
        "url": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    },
    {
        "name": "HDFC ELSS Tax Saver Fund - Direct Plan Growth",
        "scheme_type": "elss",
        "source_type": "scheme",
        "url": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    },
    {
        "name": "HDFC Small Cap Fund - Direct Growth",
        "scheme_type": "small-cap",
        "source_type": "scheme",
        "url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    },
    {
        "name": "HDFC Balanced Advantage Fund - Direct Growth",
        "scheme_type": "hybrid",
        "source_type": "scheme",
        "url": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
    },
]

# ---------------------------------------------------------------------------
# Client settings
# ---------------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HTTP_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Retrieval tuning (Phase 5 / Phase 6)
# ---------------------------------------------------------------------------
RETRIEVER_TOP_K = 5          # number of candidate chunks to return
RETRIEVER_MIN_SCORE = 0.05   # below this similarity -> no-source refusal path
SELECTION_TOP_K = 25         # broader window used by Generation for synthesis

# ---------------------------------------------------------------------------
# Generation / Answer Synthesis (Phase 7)
# ---------------------------------------------------------------------------
MAX_ANSWER_SENTENCES = 3
# Official investor-education link used when refusing opinion/advice or
# no-source questions (per PRD: "relevant educational link").
EDUCATIONAL_LINK = "https://www.amfiindia.com/investor-corner"

# ---------------------------------------------------------------------------
# Embedding (Phase 3) — Google Gemini via google-genai SDK
# ---------------------------------------------------------------------------
# gemini-embedding-001 (stable, text-only) supports task_type, so documents use
# RETRIEVAL_DOCUMENT and queries use RETRIEVAL_QUERY (the recommended RAG
# pattern). Output dim 768 is the recommended default for that model.
GEMINI_EMBEDDING_MODEL_ENV = "GEMINI_EMBEDDING_MODEL"
EMBEDDING_MODEL = os.environ.get(GEMINI_EMBEDDING_MODEL_ENV,
                                 "gemini-embedding-001")
EMBEDDING_DIM = int(os.environ.get("GEMINI_EMBEDDING_DIM", "768"))

# Optional LLM backend. The default is Google Gemini via google-genai SDK.
# Activates when GEMINI_API_KEY is set; otherwise retrieval/embedding is
# unavailable (no offline fallback without real embeddings).
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_MODEL"
GEMINI_BASE_URL_ENV = "GEMINI_BASE_URL"
LLM_MODEL = os.environ.get(GEMINI_MODEL_ENV, "gemini-3.6-flash")
LLM_BASE_URL = os.environ.get(GEMINI_BASE_URL_ENV,
                              "https://generativelanguage.googleapis.com/v1beta")
LLM_API_KEY_ENV = GEMINI_API_KEY_ENV
LLM_TIMEOUT_SECONDS = 120
LLM_TEMPERATURE = 0.4


def ensure_dirs() -> None:
    """Create all data directories if missing."""
    for d in (RAW_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, CHROMA_DIR):
        os.makedirs(d, exist_ok=True)
