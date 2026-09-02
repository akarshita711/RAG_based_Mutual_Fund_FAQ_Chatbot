# HDFC Mutual Fund FAQ Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers factual questions about HDFC Mutual Fund schemes listed on [Groww](https://groww.in/). It is strictly a **facts-only** assistant — it never gives investment advice, never computes returns, and never accepts personal information.

---

## Features

- Answers factual queries: expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, statements
- Every answer includes **one official citation link** to the Groww scheme page
- Responses capped at 3 sentences with a "Last updated from sources" timestamp
- Politely refuses opinion/portfolio questions (buy, sell, invest, compare) with an [AMFI educational link](https://www.amfiindia.com/investor-corner)
- Refuses PII (PAN, Aadhaar, OTPs, account numbers, emails, phone numbers)
- No performance claims or return computations
- Streamlit web UI with Groww-branded color palette

### In-Scope Schemes

| # | Scheme | Category |
|---|--------|----------|
| 1 | HDFC Large Cap Fund - Direct Growth | Large-cap |
| 2 | HDFC Flexi Cap Fund - Direct Growth | Flexi-cap |
| 3 | HDFC ELSS Tax Saver Fund - Direct Plan Growth | ELSS |
| 4 | HDFC Small Cap Fund - Direct Growth | Small-cap |
| 5 | HDFC Balanced Advantage Fund - Direct Growth | Hybrid |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Embeddings | `gemini-embedding-001` (Google Gemini, 768-dim) |
| LLM | `gemini-3.6-flash` (Google Gemini) |
| Vector Store | ChromaDB (local, persistent) |
| Web UI | Streamlit |
| HTTP / Parsing | Python stdlib (`urllib`, `html.parser`) |

> Zero-extra-dependency design: the only runtime dependencies are `google-genai`, `chromadb`, and `streamlit`.

---

## Project Structure

```
RAG_based_Mutual_Fund_FAQ_Chatbot/
├── .env                         # Gemini API credentials
├── .gitignore                   # Ignores .env, __pycache__/, .streamlit/secrets.toml
├── requirements.txt             # Python dependencies
│
├── .streamlit/
│   └── config.toml              # Streamlit theme (Groww green)
│
├── docs/
│   ├── problemstatement.txt     # Challenge brief and seed URLs
│   ├── PRD.md                   # Product Requirements Document
│   └── architecture.md          # Architecture & data flow docs
│
├── src/
│   ├── config.py                # Central config, paths, constants, .env loader
│   ├── gemini.py                # Google Gemini SDK wrapper
│   │
│   ├── ingest/                  # OFFLINE pipeline (Phases 1-4)
│   │   ├── loader.py            #   Phase 1: HTTP fetch, HTML-to-text, JSONL store
│   │   ├── chunker.py           #   Phase 2: Sliding-window chunking
│   │   ├── embedder.py          #   Phase 3: Gemini embedding adapter
│   │   └── build_index.py       #   Phase 4: ChromaDB index builder
│   │
│   ├── query/                   # ONLINE pipeline (Phases 5-7)
│   │   ├── retriever.py         #   Phase 5: Cosine similarity search
│   │   ├── generator.py         #   Phase 7: Prompt building, LLM synthesis
│   │   └── app.py               #   Streamlit web UI
│   │
│   └── eval/
│       └── eval_retrieval.py    # Placeholder for Phase 6
│
├── data/
│   ├── raw/                     # Scraped documents & HTML cache
│   ├── chunks/                  # Chunked records (chunks.jsonl)
│   └── vector_db/chroma/        # Persistent ChromaDB index
│
└── tests/                       # Test directory
```

---

## Architecture

The pipeline is split into two runtime flows across 7 phases:

### Offline — Ingestion (Phases 1–4)

```
URLs ──► Fetch & Cache ──► HTML→Text ──► Chunk ──► Embed ──► ChromaDB
              (Phase 1)       (Phase 1)   (Phase 2)  (Phase 3)    (Phase 4)
```

1. **Data Loading** — Fetches Groww scheme pages, converts HTML to clean text, caches raw HTML, and writes `documents.jsonl`
2. **Chunking** — Sliding-window chunker with sentence/paragraph-aware boundaries (1200 chars, 150 overlap)
3. **Embedding** — `gemini-embedding-001` via ChromaDB adapter with batch pacing and 429 retry backoff
4. **Vector Store** — ChromaDB `PersistentClient` with cosine similarity, collection `hdfc_mf_faq`

### Online — Query (Phases 5–7)

```
User Query ──► PII/Opinion Check ──► Retrieve ──► Generate ──► Answer + Citation
                  (Phase 7)           (Phase 5)    (Phase 7)
```

5. **Retrieval** — ChromaDB cosine search, top-k with minimum score threshold (0.05)
6. **Generation** — Gemini LLM with structured system prompt, scheme-aware citation selection, local fallback on API failure

### Key Safeguards

- **PII detection** blocks PAN, Aadhaar, OTPs, passwords via regex
- **Opinion detection** refuses advice-related questions (buy/sell/invest/compare) with AMFI link
- **Citation enforcement** — post-processing guarantees every answer has exactly one official source link
- **LLM fallback** — `LocalGenerator` (deterministic keyword extraction) activates on API failure

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Google Gemini API key](https://aistudio.google.com/apikey)

### 1. Clone & Install

```bash
git clone <repo-url>
cd RAG_based_Mutual_Fund_FAQ_Chatbot
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-3.6-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### 3. Build the Index

```bash
# Phase 1: Fetch & cache source pages
python -m src.ingest.loader --force

# Phase 2: Chunk the documents
python -m src.ingest.chunker --force

# Phase 3: Verify embeddings work
python -m src.ingest.embedder "test sentence"

# Phase 4: Build ChromaDB vector index
python -m src.ingest.build_index --force
```

### 4. Run the Chatbot

```bash
streamlit run src/query/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## CLI Smoke Tests

Each phase has its own CLI entry point for manual verification:

```bash
python -m src.ingest.loader [--force] [--no-cache]     # Phase 1
python -m src.ingest.chunker [--force] [--chunk-size N] # Phase 2
python -m src.ingest.embedder "text to embed"            # Phase 3
python -m src.ingest.build_index [--force]               # Phase 4
python -m src.query.retriever "query" [--k N]            # Phase 5
python -m src.query.generator "question" [--force-llm]   # Phase 7
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Google Gemini API key (required) |
| `GEMINI_MODEL` | `gemini-3.6-flash` | LLM model for generation |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `GEMINI_EMBEDDING_DIM` | `768` | Vector dimensionality |
| `GEMINI_EMBED_BATCH_SIZE` | `8` | Embedding batch size |
| `GEMINI_EMBED_BATCH_SLEEP` | `3` | Seconds between batches |

| Constant | Value | Description |
|----------|-------|-------------|
| `RETRIEVER_TOP_K` | `5` | Default retrieval candidates |
| `RETRIEVER_MIN_SCORE` | `0.05` | Minimum similarity score |
| `MAX_ANSWER_SENTENCES` | `3` | Answer length cap |
| `LLM_TEMPERATURE` | `0.4` | Generation temperature |
| `DEFAULT_CHUNK_SIZE` | `1200` | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `150` | Overlap between chunks |

---

## Sample Questions

- "What is the expense ratio of HDFC Large Cap Fund?"
- "What is the ELSS lock-in period?"
- "Is there an exit load on HDFC Small Cap Fund?"
- "What is the benchmark index for HDFC Balanced Advantage Fund?"
- "What is the riskometer rating of HDFC Small Cap Fund?"
- "How to download an account statement?"

---

## License

This project is for educational purposes.
