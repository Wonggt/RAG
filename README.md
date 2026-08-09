# 🧠 Agentic RAG — Multilingual Q&A over your documents

An **Agentic Retrieval-Augmented Generation** system built with LangGraph, LangChain,
FAISS, BM25, Cohere `embed-multilingual-v3.0`, and a BGE cross-encoder reranker.

**Live demo:** https://ragproject-andrewllm.streamlit.app

Upload PDFs / DOCX / TXT / URLs → ask questions → get answers **with inline citations**
and a **live trace of how the agent reasoned**.

---

## 🎥 What "Agentic" Actually Means Here

Traditional RAG is a fixed pipeline: **retrieve → stuff → answer**. If the retriever
returns bad chunks, the LLM answers with bad context — full stop.

**Agentic RAG** makes the LLM a decision-maker at multiple points:

| Traditional RAG | Agentic RAG (this project) |
|---|---|
| Fixed linear chain | State machine with conditional branches |
| Single retrieval pass | Retrieval can be repeated if quality is poor |
| Query used as-is | Query is **rewritten** for retrieval optimality |
| No self-check | **Grader** node evaluates retrieved docs |
| No source attribution | **Every claim carries `[N]` citations** |
| No confidence signal | Retry counter + trace shown to user |
| Bad chunks → bad answer | Bad chunks → retry with new query |

---

## 🧩 Architecture

```
                     ┌────────────────────┐
                     │   User Question    │
                     └─────────┬──────────┘
                               ▼
                     ┌────────────────────┐
                     │  🔄 Rewrite Query  │  LLM optimizes query for retrieval
                     └─────────┬──────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │           🔀 Hybrid Retrieval                │
        │  ┌───────────────┐    ┌──────────────────┐   │
        │  │  BM25         │    │  FAISS + Cohere  │   │
        │  │  (keyword)    │    │  (semantic)      │   │
        │  └──────┬────────┘    └────────┬─────────┘   │
        │         └──── Ensemble (0.4 / 0.6) ────┘     │
        │              → top-12 candidates             │
        └──────────────────────┬───────────────────────┘
                               ▼
                     ┌────────────────────┐
                     │ 🎯 BGE Reranker    │  Cross-encoder scores (query, doc)
                     │  → top-4           │  pairs — MUCH more accurate than
                     └─────────┬──────────┘  bi-encoder cosine
                               ▼
                     ┌────────────────────┐
                     │  ✅ Grade Docs     │  LLM: are these relevant?
                     └─────────┬──────────┘
                               │
                        ┌──────┴──────┐
                        ▼             ▼
                  (retry ≤1x)     (relevant)
                        │             │
                        ▼             ▼
                ┌───────────┐  ┌───────────────────────┐
                │ Re-rewrite│  │ ✍️ Generate Answer    │
                └─────┬─────┘  │    with [N] citations │
                      │        └───────────┬───────────┘
                      └──────────────┐     │
                                     │     ▼
                                     │  ┌────────────────────┐
                                     │  │ Answer + Sources   │
                                     └──│ + Trace shown to   │
                                        │ user               │
                                        └────────────────────┘
```

---

## 🔧 Tech Stack & Why

| Layer | Choice | Rationale |
|---|---|---|
| **Orchestration** | LangGraph | State machine > brittle if/else chains. Conditional edges enable the retry loop. |
| **Vector DB** | FAISS (in-memory) | Zero-config, blazing fast for demo scale. Trade: no persistence. |
| **Dense embeddings** | Cohere `embed-multilingual-v3.0` | 1024-dim, 100+ languages, no local GPU needed. Fallback: `intfloat/multilingual-e5-base`. |
| **Sparse retrieval** | BM25 (`rank_bm25`) | Catches exact keywords/numbers/names that dense search misses. |
| **Reranker** | `BAAI/bge-reranker-base` | Cross-encoder: scores (query, doc) jointly — much more accurate than cosine on bi-encoder outputs. |
| **LLM** | Any OpenRouter model | Free tier available; switchable via dropdown. |
| **Frontend** | Streamlit | Fast iteration, great for demos. |
| **TTS** | `gTTS` | Cloud-safe (unlike edge-tts which gets IP-blocked). |

### Why *both* Cohere and local e5?
Environment-based switching (12-factor):
- `COHERE_API_KEY` present → Cohere (fast, cloud-friendly, ~10x faster indexing)
- No key → local e5-base (offline, private, ~2GB RAM)

The application code is identical for both — a single `get_embedding_model()`.

---

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create `.env` in the project root:

```bash
OPENROUTER_API_KEY=sk-or-v1-...   # required (LLM)
COHERE_API_KEY=...                # optional (embeddings; falls back to local)
```

Get keys:
- OpenRouter: https://openrouter.ai/keys (free-tier models available)
- Cohere: https://dashboard.cohere.com/api-keys (trial key, no card)

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a PDF, toggle **Use RAG** + **🧠 Agentic mode**, ask.

---

## ✅ Testing Strategy

`pytest` suite in `tests/` covers **6 dimensions** with **no network calls / no API keys**
(uses a fake LLM and deterministic hash-based embeddings).

Run:

```bash
pytest -v
```

### What each test class verifies

| Test class | What it proves |
|---|---|
| `TestRetrieval` | Vector DB actually ranks on-topic docs above off-topic ones (Precision@1 sanity) |
| `TestNodes` | Each LangGraph node (rewrite, retrieve, generate) produces the expected state shape |
| `TestEndToEnd` | Full graph run produces an answer containing the ground-truth fact + citations + full trace |
| `TestRetryLogic` | Grader saying "no" triggers **exactly one** rewrite retry, then generates anyway (bounded latency) |
| `TestCitations` | Citations are 1-indexed sequential; each carries `source`, `page`, `snippet` |
| `TestMultilingual` | Chinese query surfaces the Chinese doc (or the semantically equivalent English one) |

### Why fakes instead of real LLM calls?

- **Deterministic** — same test result every run, no flakes
- **Fast** — full suite runs in <5 seconds
- **Free** — CI can run on every push without spending API credits
- **Isolated** — tests fail on *code* bugs, not on OpenRouter downtime

Real LLM behavior is validated **manually via the Streamlit UI** — the trace panel
shows exactly what the agent decided at each step.

---

## 📊 Performance Notes

Rough numbers on Streamlit Cloud (Cohere backend, 60-page PDF):

| Operation | Time |
|---|---|
| PDF download + parse | ~1 s |
| Chunking (800 chars / 120 overlap) | ~0.3 s |
| Embedding ~150 chunks via Cohere | ~2 s |
| First query (loads reranker ~278 MB) | ~15 s |
| Subsequent queries | ~2–4 s |

### Retrieval accuracy improvements (informal)
- Fixing `chunk_size=5000` → `800` alone: significant improvement — the old size
  exceeded e5's 512-token limit, silently truncating most content.
- Adding BM25 to FAISS: better recall on queries with exact entity names / numbers.
- Adding BGE reranker: consistently promotes the truly relevant chunk to #1.

---

## 🎁 Bonus Features Implemented

- ✅ **Citations** — every claim carries `[N]` markers linked to source + page number + snippet preview
- ✅ **Optimized retrieval** — Hybrid Search (BM25 + FAISS) → BGE cross-encoder rerank
- ✅ **Multilingual** — Cohere v3 covers 100+ languages out of the box; test suite verifies EN + ZH
- ✅ **Text-to-Speech** — answers can be listened to via gTTS (multilingual)
- ✅ **Live decision trace** — user sees every step the agent took (query rewrite, grader verdict, retry decision)
- ✅ **Graceful degradation** — grader failures fall back to "assume relevant"; TTS failures show a subtle notice; missing Cohere key falls back to local model

---

## 📁 Project Layout

```
.
├── app.py              # Streamlit UI + routing (plain / classic RAG / Agentic RAG)
├── agentic_rag.py      # LangGraph pipeline: rewrite → retrieve → grade → generate
├── rag_methods.py      # Document loading, embeddings, hybrid retriever, reranker
├── tts.py              # gTTS wrapper with language auto-detection
├── tests/
│   ├── conftest.py     # Fixtures: fake LLM, hash embeddings, sample corpus
│   └── test_agentic_rag.py  # 6 test classes
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 🎤 Discussion — Thought Process

1. **Started with classic RAG** — LangChain's `create_retrieval_chain` gave a fixed pipeline. Fast to build, but poor recovery: bad retrieval → bad answer.
2. **Identified failure modes** during manual testing: vague user queries under-retrieved; keyword-heavy queries (numbers, proper nouns) missed by pure semantic search; no way to tell if the answer was grounded.
3. **Moved to LangGraph** to add decision points: query rewriting for retrieval optimality, a grader node for self-correction, and citations for user trust.
4. **Layered in optimized retrieval** — hybrid search + reranker — because the grader can only work with what retrieval surfaces.
5. **Switched embeddings from local e5 to Cohere API** on the deployed instance for order-of-magnitude faster indexing on Streamlit Cloud's CPU-only workers, keeping the local model as an offline fallback.
6. **Wrote tests** that exercise each node in isolation *and* the end-to-end graph, using fakes so the suite is deterministic and free to run.

### Trade-offs I consciously made

| Chose | Instead of | Why |
|---|---|---|
| In-memory FAISS | Persistent Pinecone / Weaviate | Zero infra for a demo; would swap for production. |
| Max 1 retry in grader loop | Unbounded retries | Bounded latency; a stubborn grader shouldn't hang the UI. |
| Fake-LLM tests | Real-LLM eval suite | Deterministic & free CI; real evaluation happens via UI trace panel. |
| gTTS | edge-tts | Streamlit Cloud IPs get 403'd by Microsoft's endpoint. |

---

## 📄 License

MIT
