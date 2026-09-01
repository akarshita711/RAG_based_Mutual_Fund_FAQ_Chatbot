# Architecture — HDFC Mutual Fund FAQ Chatbot (Groww)

**Author:** Senior Architect · **Version:** 1.0
**Scope:** Restricted to `docs/PRD.md` only (small RAG prototype, facts-only, one citation per answer).

---

## 1. Design Principles

- **Small & simple** — 15–25 page corpus, single prototype; no over-engineering.
- **Citation integrity** — every answer must map to a retrieved source (deterministic link).
- **Facts-only** — published sources only; safe-refusal for opinion/portfolio questions.
- **PII-safe** — never persist user input.
- **Phased & testable** — each pipeline stage is independently verifiable.

## 2. High-Level Pipeline

```
[1 Data Loading] → [2 Chunking] → [3 Embedding] → [4 Vector Store]
                                                          │
User Query ──────────────────────────────────────────────► [5 Retrieval Logic]
                                                          │        │
                                                          ▼        ▼
                                            [6 Retrieval Testing]  │
                                                          │        │
                                                          └──▶ [7 Generation / Answer Synthesis] → Response (+1 citation)
```

Interpreted in two runtime flows:
- **Ingest (offline):** Phases 1 → 4 build the index.
- **Query (online):** Phases 5 → 7 answer the user.

## 3. Phases

### Phase 1 — Data Loading
- **Input:** 15–25 URLs from Groww/AMC/SEBI/AMFI (factsheets, KIM/SID, scheme FAQs, fees/charges, riskometer/benchmark, tax/statement guides). Core = the 5 HDFC scheme pages in the PRD.
- **Tasks:**
  - Fetch each page (HTTP + HTML parsing).
  - Extract readable text/structured tables (expense ratio, exit load, min SIP, lock-in, riskometer, benchmark).
  - Persist raw page text + metadata: `url`, `title`, `source_type`, `fetched_at`.
- **Output:** raw document store (JSONL/SQLite).

### Phase 2 — Chunking
- **Tasks:**
  - Split each cleaned page into overlapping text chunks (e.g., 300–500 tokens, small overlap).
  - Preserve source metadata (`url`, `title`) on every chunk.
- **Why:** facts like "expense ratio" should stay within a single retrievable chunk.
- **Output:** chunk store with per-chunk metadata.

### Phase 3 — Embedding
- **Tasks:**
  - Choose an embedding model (e.g., a lightweight open/built-in model).
  - Encode every chunk into a vector.
- **Output:** chunk_id → embedding vector mapping.

### Phase 4 — Vector Store
- **Tasks:**
  - Persist chunk vectors with metadata into a simple vector index (in-memory / small DB).
  - Support similarity search (cosine / dot product).
- **Output:** searchable index for retrieval.

### Phase 5 — Retrieval Logic
- **Tasks:**
  - Encode the user query into a vector.
  - Retrieve **top-k** most similar chunks.
  - If no chunk scores above a minimum threshold → **no-source refusal path**.
- **Output:** ranked list of candidate chunks + their source URLs.

### Phase 6 — Retrieval Testing
- **Tasks (offline, quality gate):**
  - Run a set of factual queries against retrieval.
  - Verify the correct scheme/source is returned for each query type (expense ratio, ELSS lock-in, min SIP, exit load, "Very High risk", benchmark, statements).
    - Note: Groww pages phrase the riskometer as "Very High risk"; the word "riskometer" does not appear in the corpus, so queries must use "Very High risk" (or be refused with an educational link).
  - Measure recall@k / hit-rate; tune chunk size, overlap, and k.
- **Gate:** only proceed to Phase 7 once factual queries reliably surface the right source.

### Phase 7 — Generation / Answer Synthesis
- **Tasks:**
  - Build LLM prompt (W1/W2) with the retrieved chunks as context.
  - Decide **answer vs. safe-refusal**:
    - Factual, in-corpus → synthesize answer (≤3 sentences) + exactly **one citation link**.
    - Opinion/portfolio (buy/sell) → politely refuse + one official educational link.
    - No-source fact → refuse + educational link.
  - Enforce constraints: ≤3 sentences, append "Last updated from sources: <date>.", no returns computed, no PII stored.
- **Output:** final response with citation (UI = welcome line + 3 examples + "Facts-only. No investment advice.").

## 4. Component / Technology Mapping (simple)

| Phase | Component | Simple choice |
|---|---|---|
| 1 | Loader/Parser | HTTP + HTML parser (e.g., BeautifulSoup) |
| 2 | Chunker | Rule-based token splitter with overlap |
| 3 | Encoder | Lightweight embedding model |
| 4 | Vector store | In-memory / small local index |
| 5 | Retriever | Top-k similarity search |
| 6 | Test harness | Script/notebook + eval queries |
| 7 | LLM | Facts-only prompt path |

## 5. Data Flow & Metadata

- Every chunk carries: `chunk_id`, `url`, `title`, `source_type`, `fetched_at`.
- Every answer carries its citation from the retrieved chunk's `url`.
- No user-query or PII is persisted (constraint FR-5 / NFR-2).

## 6. Constraints from PRD (mapped)

| PRD item | Where honored |
|---|---|
| FR-2 one citation link | Phase 7 |
| FR-3 ≤3 sentences + "Last updated" | Phase 7 |
| FR-4 refuse opinion | Phase 7 |
| FR-5 no PII | Phase 1 (schema) + Phase 7 |
| FR-6 no returns | Phase 7 |
| NFR-2 public sources only | Phase 1 |
| NFR-4 deterministic citations | Phase 6 + 7 |

## 7. Risks (architectural)

- Groww markup drift → wrap Phase 1 parser in a thin adapter, treat failures as skipped/refused pages.
- Legacy "equity fund" slug for flexi-cap → keep canonical scheme→URL mapping.
- Small corpus gaps → Phase 6 surfaces coverage; Phase 7 refuses cleanly.
- Freshness (expense ratio changes) → store `fetched_at`, surface in "Last updated" line.

## 8. Out of Scope (per PRD §12)

- Login/account, user history/personas, financial advice engine.
- External hosting beyond prototype/demo.
