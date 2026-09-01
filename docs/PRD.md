# PRD — HDFC Mutual Fund FAQ Chatbot (Groww)

**Author:** PM · **Status:** Draft · **Version:** 1.0
**Source:** `docs/problemstatement.txt`

---

## 1. Problem Statement

Retail users comparing mutual fund schemes (and support/content teams answering repetitive questions) need a fast, trustworthy source for **facts-only** answers about HDFC schemes on Groww — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements. Today these facts are scattered across official pages, and generic chatbots give answers without citations or give advice. We want a **small RAG chatbot** that answers only factual questions and always shows one official source link.

## 2. Goals & Non-Goals

### Goals
- Answer factual FAQ queries about the 5 chosen HDFC schemes with one citation link each.
- Refuse opinion/portfolio questions politely and point to an official educational link.
- Keep answers concise (≤3 sentences) with a "Last updated from sources" note.
- Tiny, simple UI: welcome line + 3 example questions + "Facts-only. No investment advice."

### Non-Goals
- No investment advice, buy/sell recommendations, or personal portfolio help.
- No return/performance computation or comparison.
- No POS/backend screenshots in sources; public sources only.

## 3. Scope

**Product:** Groww (`https://groww.in/`)
**AMC:** HDFC
**Schemes (5 pages, RAG corpus seed):**

| Type | Scheme | URL |
|---|---|---|
| Large-cap | HDFC Large Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| Flexi-cap | HDFC Flexi Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| ELSS | HDFC ELSS Tax Saver Fund – Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| Small-cap | HDFC Small Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Hybrid | HDFC Balanced Advantage Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

**Corpus target:** 15–25 public pages from AMC/SEBI/AMFI/Groww (factsheets, KIM/SID, scheme FAQs, fees/charges, riskometer/benchmark notes, tax/statement guides). Includes the 5 scheme pages above as the core.

## 4. Target User & Use Cases

- **Retail users:** comparing schemes, checking fees, lock-in, minimum SIP, risk.
- **Support/content teams:** answering repetitive MF questions quickly and consistently.

Example queries (must answer):
- "What is the expense ratio of HDFC Large Cap Fund?"
- "What is the ELSS lock-in period?"
- "What is the minimum SIP amount?"
- "Is there an exit load?"
- "What is the riskometer / benchmark?" → queried as "Is the risk Very High?" / "What is the benchmark?" (Groww pages phrase risk as "Very High risk"; the literal word "riskometer" is not on the pages)
- "How do I download a capital-gains statement?"

Must refuse:
- "Should I buy or sell this fund?" (opinion/portfolio)

## 5. Functional Requirements

- **FR-1** Answer factual scheme questions using retrieval from the corpus.
- **FR-2** Every answer includes **exactly one clear citation link** to the official source.
- **FR-3** Answers ≤3 sentences; end with "Last updated from sources: <date>."
- **FR-4** Politely refuse opinion/portfolio questions with a facts-only message + one educational link.
- **FR-5** Do not accept or store PII (PAN, Aadhaar, account numbers, OTPs, emails, phone numbers).
- **FR-6** No return/performance computation; link to official factsheet if asked about returns.
- **FR-7** UI shows welcome line, 3 example questions, and note "Facts-only. No investment advice."

## 6. Non-Functional Requirements

- **NFR-1** Simple, runnable prototype (app or notebook).
- **NFR-2** Public sources only; no third-party blogs.
- **NFR-3** Low latency for small corpus; retrieval speed sufficient for a demo.
- **NFR-4** Deterministic citations — answer must map to the retrieved source.

## 7. Architecture (High Level)

```
User Query
   │
   ▼
[RAG Prototype]
   ├─ Embedding/index of corpus (15–25 pages)
   ├─ Retrieve top-k relevant chunks
   ├─ LLM prompt: facts-only, ≤3 sentences, citation
   └─ Decide: answer vs. safe-refusal
   │
   ▼
[Response: answer + 1 citation link]
```

- **Corpus:** scraped/parsed from official pages (factsheets, KIM/SID, FAQs).
- **Index:** chunked embeddings for retrieval (W3).
- **LLM:** instruction-following prompt (W2) that reasons about the exact fact (W1).

## 8. Skills Being Tested

- **W1 — Thinking Like a Model:** identify the exact fact asked; decide answer vs. refuse.
- **W2 — LLMs & Prompting:** concise phrasing, polite safe-refusals, citation wording.
- **W3 — RAGs:** small-corpus retrieval with accurate citations.

## 9. Deliverables

1. Working prototype (app or notebook) OR ≤3-min demo video.
2. Source list (CSV/MD) of 15–25 URLs.
3. README — setup steps, scope (AMC + schemes), known limits.
4. Sample Q&A file — 5–10 queries with answers + links.
5. Disclaimer snippet used in UI.

## 10. Success Metrics (prototype)

- 100% of factual test queries return an answer **with a valid citation link**.
- 100% of opinion/portfolio test queries are **politely refused** (no direct advice).
- 0 answers missing "Last updated from sources" line.
- Answers ≤3 sentences for ≥90% of cases.

## 11. Risks / Known Limits

- Groww page markup changes may affect scraping (note: flexi-cap uses legacy "equity fund" slug).
- Some facts (e.g., expense ratio) update periodically; must re-validate freshness.
- Small corpus limits coverage — some questions may have no source → must refuse and point to education link.
- Hosting a live app may not be possible → demo video fallback.

## 12. Out of Scope

- Login/account features.
- User history/personas.
- Actual financial advice engine.
