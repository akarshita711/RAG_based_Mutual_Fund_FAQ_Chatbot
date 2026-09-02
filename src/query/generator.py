"""Phase 7 — Generation / Answer Synthesis.

Takes the user's question and the retrieved chunks, and produces a facts-only
answer:

- Answers ONLY from the provided context.
- Always includes exactly one source link (citation).
- Keeps answers to <= ``MAX_ANSWER_SENTENCES`` sentences.
- Appends "Last updated from sources: <date>."
- Politely refuses opinion/advice/portfolio questions (e.g., "Should I buy/sell?")
  with a facts-only message and an official educational link.
- Refuses when no retrieved chunk supports the question (no-source path).

Backends behind one facade:

1. **LLM backend** (``GeminiGenerator``): calls Gemini via the google-genai
   SDK (``generate_text`` in src/gemini.py). Requires ``GOOGLE_API_KEY``.
2. **Local backend** (``LocalGenerator``): deterministic extraction fallback
   used only if the LLM call fails (network/API error).

Note: retrieval itself uses Gemini query embeddings, so a missing API key
surfaces at Phase 5 as ``embedding_unavailable`` and Generation refuses with
an educational link.

Usage:
    from src.query.generator import generate
    result = generate("What is the ELSS lock-in period?", retrieve=...)

Output dict shape:
    {
      "question", "answer", "answerable": bool,
      "citation": {"url", "title", "fetched_at"} | None,
      "refused": bool, "refusal_reason": str | None,
      "educational_link": str | None,
      "last_updated": str | None,   # "YYYY-MM-DD" from fetched_at
    }
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from src.gemini import LLMError, generate_text  # noqa: E402


# ---------------------------------------------------------------------------
# Refusal detection (W1: identify the exact intent, decide answer vs. refuse)
# ---------------------------------------------------------------------------
OPINION_PATTERN = re.compile(
    r"\b(should|would|advice|suggest|recommend|opinion|buy\b|sell|invest"
    r"\s+in|portfolio|diversif|market\s*timing|good\s+invest|worth|better\s+fund"
    r"|compare|vs\.?|versus|switch|redeem|withdraw)\b",
    re.IGNORECASE,
)
PII_PATTERN = re.compile(
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    r"|\b\d{12}\b"
    r"|\b(?:otp|aadhaar|pan|password)\b",
    re.IGNORECASE,
)


def is_opinion_question(question: str) -> bool:
    """Heuristic: does this read like opinion/advice/portfolio? (W1)"""
    return bool(OPINION_PATTERN.search(question))


def contains_pii(question: str) -> bool:
    """Reject any request that includes personal identifiers."""
    return bool(PII_PATTERN.search(question))


# ---------------------------------------------------------------------------
# Shared prompt builder (W2: instruction style, citation wording)
# ---------------------------------------------------------------------------
def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """System prompt + user message with only retrieved chunks as context."""
    system = textwrap.dedent(
        """\
        You are a facts-only mutual fund FAQ assistant for HDFC schemes on Groww.
        Follow these rules strictly:

        1. Answer ONLY from the provided context chunks. Never use outside knowledge.
        2. If the context does not contain the fact, politely refuse: say you
           can only answer facts from official pages, and give the educational
           link exactly: {edu}.
        3. If the question asks for opinion, advice, buy/sell recommendations,
           portfolio help, or personal decisions, politely refuse: explain you
           are facts-only and provide the educational link exactly: {edu}.
        4. Do not compute, compare, or predict returns/performance.
        5. Never accept, request, or repeat PII (PAN, Aadhaar, account numbers,
           OTPs, emails, phone numbers). Refuse and give the educational link.
        6. Each answer is at most {sent} sentences, ends with the exact line:
           "Last updated from sources: {date}."
        7. Every answer includes exactly one citation: "Source: <url>" where
           <url> is from the context (use the context url verbatim).

        Context chunks:
        {context}
        """
    ).format(
        edu=config.EDUCATIONAL_LINK,
        sent=config.MAX_ANSWER_SENTENCES,
        date="<fetched_date>",
        context=_format_context(chunks),
    )

    user = textwrap.dedent(
        """\
        Question: {q}

        If you answer, structure it exactly as:
        <answer, max {sent} sentences>

        Source: <one url from context>

        Last updated from sources: <fetched_date>.
        """
    ).format(q=question, sent=config.MAX_ANSWER_SENTENCES)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("source_type", "scheme")
        lines.append(f"[Chunk {i}] url={c.get('url','')} source_type={src}")
        lines.append(f"text: {c.get('text','')}")
    return "\n".join(lines)


def _best_chunk(chunks: list[dict]):
    """Highest-scoring retrieved chunk (hits are already sorted by score)."""
    return chunks[0] if chunks else None


SCHEME_FRAGMENTS = [
    ("large cap", "large-cap"),
    ("flexi cap", "flexi-cap"),
    ("elss", "elss"),
    ("small cap", "small-cap"),
    ("balanced advantage", "hybrid"),
]


def _pick_best_hit(question: str, hits: list[dict]) -> dict | None:
    """Pick the citation chunk.

    If the question names an in-scope scheme, prefer the highest-scoring hit
    from that scheme so the citation matches what the answer is about;
    otherwise fall back to the top hit.
    """
    if not hits:
        return None
    q = question.lower()
    for frag, scheme_type in SCHEME_FRAGMENTS:
        if frag in q:
            for h in hits:
                if h.get("scheme_type") == scheme_type:
                    return h
            break
    return hits[0]


def _question_terms(question: str) -> list[str]:
    """Distinctive lowercase tokens from the question (stopwords removed)."""
    stop = _STOPWORDS
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{1,}", question.lower())
    return [w for w in words if w not in stop and len(w) > 2]


_STOPWORDS = {
    "the", "and", "for", "with", "are", "you", "your", "this", "that",
    "from", "have", "has", "was", "were", "will", "can", "all", "any",
    "but", "not", "its", "also", "than", "then", "into", "over", "under",
    "what", "which", "when", "where", "how", "who", "why", "is", "of", "to",
    "in", "on", "at", "a", "an", "or", "as", "by", "it", "be", "do", "does",
    "my", "me", "does", "what", "should",
}


# ---------------------------------------------------------------------------
# LLM backend — Google Gemini via google-genai SDK (src/gemini.py)
# ---------------------------------------------------------------------------
class GeminiGenerator:
    """Gemini generateContent client built on the google-genai SDK.

    Reads GOOGLE_API_KEY / GEMINI_MODEL / GEMINI_BASE_URL from config/.env.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 timeout: int | None = None) -> None:
        self.model = model or config.LLM_MODEL
        # SDK client reads the key itself; validate presence up front.
        self.api_key = (api_key if api_key is not None
                        else os.environ.get(config.GEMINI_API_KEY_ENV, "") or "")
        self.timeout = timeout or config.LLM_TIMEOUT_SECONDS

    def complete(self, messages: list[dict]) -> str:
        """Call Gemini with OpenAI-style [system, user] messages."""
        if not self.api_key:
            raise LLMError("No Gemini API key configured (GOOGLE_API_KEY).")
        system, user = _split_messages(messages)
        return generate_text(system, user)


def _split_messages(messages: list[dict]) -> tuple[str, str]:
    """Separate OpenAI-style messages into (system, user) strings."""
    system = ""
    user = ""
    for m in messages:
        content = m.get("content", "")
        if m.get("role") == "system":
            system = system + "\n" + content if system else content
        else:
            user = user + "\n" + content if user else content
    return system, user


# ---------------------------------------------------------------------------
# Local backend (deterministic, offline)
# ---------------------------------------------------------------------------
class LocalGenerator:
    """Extracts the fact directly from the best retrieved chunk.

    Satisfies FR-2/3/4 without a network dependency; used automatically when no
    LLM endpoint is available.
    """

    # -- intent detection (W1: identify the exact fact asked) -----------------
    INTENT_KEYWORDS = {
        "expense_ratio": ["expense ratio"],
        "sip": ["min. for sip", "minimum sip", "sip amount", "sip"],
        "exit_load": ["exit load", "exit-load"],
        "lock_in": ["lock-in", "lock in", "lock"],
        "risk": ["very high risk", "very high", "riskometer", "risk"],
        "benchmark": ["benchmark"],
        "statement": ["statement", "capital gains", "gains", "dividend"],
    }

    # Question fragment -> scheme_type (for the 5 in-scope schemes).
    SCHEME_FRAGMENTS = [
        ("large cap", "large-cap"),
        ("flexi cap", "flexi-cap"),
        ("elss", "elss"),
        ("small cap", "small-cap"),
        ("balanced advantage", "hybrid"),
    ]

    def _detect_intent(self, question: str) -> str | None:
        q = question.lower()
        for intent, kws in self.INTENT_KEYWORDS.items():
            if any(k in q for k in kws):
                return intent
        return None

    def _infer_scheme(self, question: str) -> str | None:
        q = question.lower()
        for frag, scheme_type in self.SCHEME_FRAGMENTS:
            if frag in q:
                return scheme_type
        return None

    def _extract(self, chunk: dict, intent: str | None = None) -> str:
        """Best-effort: find the fact window for the detected intent.

        Anchors on the intent's own keyword(s) when known, else the earliest
        fact keyword in the chunk, else the first substantial line.
        Returns a clean, single-sentence fact (not a raw page dump).
        """
        text = chunk.get("text", "")
        # Intent-specific anchor takes priority.
        if intent:
            for kw in self.INTENT_KEYWORDS.get(intent, []):
                idx = text.lower().find(kw)
                if idx != -1:
                    return self._snippet(text, idx)
        # Generic earliest fact keyword.
        candidates = [
            "Expense ratio", "Expense Ratio", "Minimum SIP", "Min. for SIP",
            "Exit load", "Exit Load", "lock-in", "Lock-in", "Benchmark",
            "Fund benchmark", "Riskometer", "Very High risk", "Minimum Lumpsum",
        ]
        best: tuple[int, str] | None = None
        for kw in candidates:
            idx = text.lower().find(kw.lower())
            if idx == -1:
                continue
            if best is None or idx < best[0]:
                best = (idx, self._snippet(text, idx))
        if best is None:
            first = next((ln.strip() for ln in text.splitlines()
                          if len(ln.strip()) > 20), None)
            return self._finish((first or text)[:180])
        return best[1]

    @staticmethod
    def _finish(snippet: str) -> str:
        """Ensure the snippet reads as one complete sentence ending in a period."""
        snippet = snippet.strip()
        if snippet and not snippet.endswith((".", "!", "?")):
            snippet += "."
        return snippet

    @staticmethod
    def _snippet(text: str, idx: int, width: int = 140) -> str:
        """Pull a fact window from ``idx``, stopping at a clean boundary.

        Stops before a newline or a known field-label boundary so unrelated
        facts (e.g. "Expense ratio", "Fund size") don't run together.
        """
        window = text[idx: idx + width]
        boundary = re.search(
            r"\s+(?=Expense Ratio|Expense ratio|Fund size|Exit load|"
            r"Lock-in|Benchmark|Riskometer|Minimum Lumpsum|Min\. for Lumpsum|"
            r"Rating\b|AUM\b)",
            window,
        )
        if boundary:
            window = window[: boundary.start()]
        snippet = window.replace("\n", " ").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        snippet = re.sub(r"\.{4,}", " ", snippet).strip()
        return LocalGenerator._finish(snippet)

    def _select_best(self, question: str, chunks: list[dict]) -> dict | None:
        """Pick the chunk whose terms best overlap the question (W1).

        Prefer a chunk that contains the question's distinctive tokens and the
        fact keywords, over a higher-similarity but off-topic chunk.
        """
        if not chunks:
            return None

        intent = self._detect_intent(question)
        intent_kws = self.INTENT_KEYWORDS[intent] if intent else None
        scheme = self._infer_scheme(question)

        # Prefer chunks holding the exact fact; if none do, use all.
        if intent_kws:
            match = [c for c in chunks
                     if any(k in (c.get("text") or "").lower() for k in intent_kws)]
            if match:
                chunks = match

        # Within the narrowed set, rank by retrieval score then scheme match.
        def score(c: dict) -> float:
            own = float(c.get("score", 0.0))
            if scheme:
                own += 10.0 if c.get("scheme_type") == scheme else 0.0
            return own

        return max(chunks, key=score)

    def synthesize(self, question: str, chunks: list[dict],
                   last_updated: str | None) -> dict:
        intent = self._detect_intent(question)
        chunk = self._select_best(question, chunks)
        if chunk is None:
            return _refusal(
                question,
                "I can only answer facts from the official pages in my corpus, "
                "and I couldn't find a matching page for that question. "
                f"For mutual fund basics, please see: {config.EDUCATIONAL_LINK}",
                refusal_reason="no_source",
                educational_link=config.EDUCATIONAL_LINK,
            )
        fact = self._extract(chunk, intent)
        answer = f"{fact} Source: {chunk['url']}"
        if last_updated:
            answer += f" Last updated from sources: {last_updated}."
        return {
            "question": question,
            "answer": answer,
            "answerable": True,
            "citation": {
                "url": chunk.get("url", ""),
                "title": chunk.get("title", "") or chunk.get("name", ""),
                "fetched_at": chunk.get("fetched_at", ""),
            },
            "refused": False,
            "refusal_reason": None,
            "educational_link": None,
            "last_updated": last_updated,
        }


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------
def _refusal(question: str, message: str, refusal_reason: str,
             educational_link: str) -> dict:
    return {
        "question": question,
        "answer": message,
        "answerable": False,
        "citation": None,
        "refused": True,
        "refusal_reason": refusal_reason,
        "educational_link": educational_link,
        "last_updated": None,
    }


def generate(question: str, retrieve: Callable, *,
             generator=None, top_k: int | None = None,
             min_score: float | None = None) -> dict:
    """Full Phase 7 flow: classify intent, retrieve, synthesize/refuse.

    ``retrieve`` is a callable matching ``Retriever.retrieve(query, k,
    min_score)``. ``generator`` optional; defaults to an LLM generator if an
    API key is present, else the LocalGenerator.

    Generation requests a broader candidate window (``SELECTION_TOP_K``) than
    the default retriever ``top_k`` so the fact-bearing chunk is present for
    synthesis; the final citation points to the single best chunk.
    """
    # 1) Intent + safety checks (W1).
    if contains_pii(question):
        return _refusal(
            question,
            "I can't accept personal identifiers. Please don't share PAN, "
            "Aadhaar, account numbers, OTPs, emails, or phone numbers. "
            f"For mutual fund basics, please see: {config.EDUCATIONAL_LINK}",
            refusal_reason="pii",
            educational_link=config.EDUCATIONAL_LINK,
        )
    if is_opinion_question(question):
        return _refusal(
            question,
            "I'm a facts-only assistant and can't give investment advice or "
            "buy/sell opinions. "
            f"For mutual fund education, please see: {config.EDUCATIONAL_LINK}",
            refusal_reason="opinion",
            educational_link=config.EDUCATIONAL_LINK,
        )

    # 2) Retrieve candidates (broad window for synthesis).
    context_k = top_k if top_k else config.SELECTION_TOP_K
    try:
        ret = retrieve(question, k=context_k, min_score=min_score)
    except TypeError:
        # Fallback for retrievers without kw overrides.
        ret = retrieve(question)
    hits = ret.get("hits", [])
    if ret.get("refused") or not hits:
        reason = ret.get("reason", "no_source")
        if reason in ("embedding_unavailable", "vector_store_unavailable"):
            message = (
                "I couldn't run the semantic search for that question right now "
                f"(retrieval unavailable: '{reason}'). Make sure GOOGLE_API_KEY "
                "is set and the vector index has been built "
                "(python -m src.ingest.build_index). "
                f"For mutual fund basics, please see: {config.EDUCATIONAL_LINK}"
            )
        else:
            message = (
                "I can only answer facts from the official pages in my corpus, "
                "and I couldn't find a matching page for that question. "
                f"For mutual fund basics, please see: {config.EDUCATIONAL_LINK}"
            )
        return _refusal(
            question,
            message,
            refusal_reason=reason,
            educational_link=config.EDUCATIONAL_LINK,
        )

    # 3) Last-updated date from the best chunk (scheme-aware citation).
    best = _pick_best_hit(question, hits)
    fetched = (best.get("fetched_at") or "")[:10] if best else ""
    last_updated = fetched or None

    # 4) Synthesize via chosen backend.
    if generator is None:
        generator = _default_generator()
    if isinstance(generator, GeminiGenerator):
        try:
            messages = build_messages(question, hits)
            raw = generator.complete(messages)
            return _normalize_llm_result(question, raw, best, last_updated)
        except LLMError as exc:
            print(f"[generator] Gemini unavailable ({exc}); falling back to local",
                  file=sys.stderr)
            generator = LocalGenerator()
    return generator.synthesize(question, hits, last_updated)


def _default_generator():
    key = os.environ.get(config.GEMINI_API_KEY_ENV, "")
    if key:
        return GeminiGenerator(api_key=key)
    return LocalGenerator()


def _normalize_llm_result(question: str, raw: str, best: dict,
                          last_updated: str | None) -> dict:
    """Shape the LLM's free-form answer into the standard result dict.

    Enforces the citation and last-updated line deterministically (belt and
    suspenders over the prompt), and keeps <= answer-sentence cap best-effort.
    """
    raw = (raw or "").strip()
    # Enforce one citation using the best chunk's URL.
    url = (best.get("url") or "").lower()
    has_cit = bool(url) and url in raw.lower()
    if not has_cit and best.get("url"):
        raw = f"{raw}\n\nSource: {best['url']}"
    # Enforce last-updated line (only if the model didn't already add one).
    if last_updated and "last updated from sources" not in raw.lower():
        raw = f"{raw}\n\nLast updated from sources: {last_updated}."
    return {
        "question": question,
        "answer": raw,
        "answerable": True,
        "citation": {
            "url": best.get("url", ""),
            "title": best.get("title", "") or best.get("name", ""),
            "fetched_at": best.get("fetched_at", ""),
        },
        "refused": False,
        "refusal_reason": None,
        "educational_link": None,
        "last_updated": last_updated,
    }


def main() -> None:
    """CLI smoke harness: python -m src.query.generator "question text"."""
    import argparse

    from src.query.retriever import Retriever  # local import, optional dep

    parser = argparse.ArgumentParser(description="Phase 7 — Generation")
    parser.add_argument("question", help="user question")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--force-llm", action="store_true",
                        help="use LLM backend if key set; else fail")
    args = parser.parse_args()

    print("question:", args.question)

    retriever = Retriever(top_k=args.k, min_score=args.min_score)
    if args.force_llm:
        if not os.environ.get(config.GEMINI_API_KEY_ENV):
            print("[generator] --force-llm requested but no GOOGLE_API_KEY set.")
            sys.exit(1)
        gen = GeminiGenerator()
    else:
        gen = None  # auto

    result = generate(args.question, retriever.retrieve, generator=gen,
                      top_k=args.k, min_score=args.min_score)

    print("refused:", result["refused"],
          "| answerable:", result["answerable"],
          "| reason:", result.get("refusal_reason"))
    print("-" * 60)
    print(result["answer"])
    if result["educational_link"]:
        print("edu link:", result["educational_link"])
    if result["citation"]:
        print("citation:", result["citation"]["url"])
    print("last_updated:", result["last_updated"])


if __name__ == "__main__":
    main()