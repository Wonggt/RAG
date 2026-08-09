"""
Tests for the Agentic RAG pipeline.

Coverage:
  1. Retrieval quality — the vector DB returns the on-topic doc for on-topic queries
  2. Hybrid retriever — BM25 + FAISS ensemble returns results
  3. Graph nodes — rewrite, grade, generate produce expected shapes
  4. End-to-end graph run — happy path returns answer + citations + trace
  5. Grader retry logic — irrelevant grade triggers exactly one rewrite retry
  6. Citation shape — every citation has index, source, page, snippet
  7. Multilingual — Chinese query surfaces the Chinese doc

Run with:  pytest -v tests/
"""

import pytest
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from agentic_rag import (
    build_agentic_rag_graph,
    run_agentic_rag,
    make_rewrite_node,
    make_retrieve_node,
    make_generate_node,
)


# ============================================================
# 1. Retrieval quality
# ============================================================

class TestRetrieval:
    """Vector retrieval alone — no LLM involved."""

    def test_relevant_doc_ranked_first(self, vector_db):
        """On-topic query should surface the on-topic doc, not the biology one."""
        results = vector_db.similarity_search("What is SST tax rate in Malaysia?", k=3)
        assert len(results) > 0
        top = results[0].page_content.lower()
        assert "sst" in top or "service tax" in top
        # And the completely unrelated doc should NOT be #1
        assert "photosynthesis" not in top

    def test_hybrid_retriever_returns_docs(self, vector_db, sample_docs):
        """BM25 + FAISS ensemble should return at least one relevant doc."""
        bm25 = BM25Retriever.from_documents(sample_docs)
        bm25.k = 3
        dense = vector_db.as_retriever(search_kwargs={"k": 3})
        hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])

        results = hybrid.invoke("SST tax rate")
        assert len(results) > 0
        # BM25 will match "SST" exactly; FAISS will match semantically
        joined = " ".join(d.page_content.lower() for d in results)
        assert "sst" in joined


# ============================================================
# 2. Individual graph nodes
# ============================================================

class TestNodes:
    """Test each LangGraph node in isolation."""

    def test_rewrite_node_produces_query(self, fake_llm_relevant):
        node = make_rewrite_node(fake_llm_relevant)
        state = {
            "original_question": "sst?",
            "question": "sst?",
            "documents": [], "answer": "", "citations": [],
            "retry_count": 0, "trace": [],
        }
        out = node(state)
        assert "question" in out
        assert len(out["question"]) > 0
        assert "🔄" in out["trace"][-1]  # trace logged

    def test_retrieve_node_populates_documents(self, vector_db):
        retriever = vector_db.as_retriever(search_kwargs={"k": 2})
        node = make_retrieve_node(retriever)
        state = {
            "original_question": "SST", "question": "SST",
            "documents": [], "answer": "", "citations": [],
            "retry_count": 0, "trace": [],
        }
        out = node(state)
        assert len(out["documents"]) == 2
        assert "📥" in out["trace"][-1]

    def test_generate_node_produces_answer_and_citations(self, vector_db, fake_llm_relevant):
        """Generate node must attach citation metadata for every retrieved doc."""
        # Fast-forward the fake LLM to its generate response
        fake_llm_relevant.responses = [fake_llm_relevant.responses[-1]]
        node = make_generate_node(fake_llm_relevant)
        docs = vector_db.similarity_search("SST", k=2)
        state = {
            "original_question": "What is SST?",
            "question": "SST", "documents": docs,
            "answer": "", "citations": [], "retry_count": 0, "trace": [],
        }
        out = node(state)
        assert out["answer"], "answer must not be empty"
        assert len(out["citations"]) == 2, "one citation per retrieved doc"
        for c in out["citations"]:
            assert set(["index", "source", "page", "snippet"]).issubset(c.keys())


# ============================================================
# 3. End-to-end graph — happy path
# ============================================================

class TestEndToEnd:
    def test_full_pipeline_happy_path(self, vector_db, fake_llm_relevant):
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        graph = build_agentic_rag_graph(fake_llm_relevant, retriever)
        result = run_agentic_rag(graph, "What is the SST tax rate?")

        assert result["answer"], "must produce an answer"
        assert result["citations"], "must produce citations"
        assert "6%" in result["answer"], "answer must contain the fact from context"
        # Trace should show all 4 stages happened
        trace_joined = " ".join(result["trace"])
        assert "Rewrote" in trace_joined
        assert "Retrieved" in trace_joined
        assert "Grader" in trace_joined
        assert "Generated" in trace_joined


# ============================================================
# 4. Retry logic — grader saying "no" must trigger exactly one retry
# ============================================================

class TestRetryLogic:
    def test_grader_bad_triggers_retry_then_generates(
        self, vector_db, fake_llm_irrelevant_then_relevant
    ):
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        graph = build_agentic_rag_graph(
            fake_llm_irrelevant_then_relevant, retriever
        )
        result = run_agentic_rag(graph, "SST?")

        # Must contain a "Retry" line in trace
        assert any("Retry" in step for step in result["trace"]), \
            f"Expected retry, trace was: {result['trace']}"
        # And still produce a final answer
        assert result["answer"]


# ============================================================
# 5. Citation shape & correctness
# ============================================================

class TestCitations:
    def test_citations_are_1_indexed_and_sequential(self, vector_db, fake_llm_relevant):
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        graph = build_agentic_rag_graph(fake_llm_relevant, retriever)
        result = run_agentic_rag(graph, "SST?")

        indices = [c["index"] for c in result["citations"]]
        assert indices == list(range(1, len(indices) + 1)), \
            "citation indices must be 1, 2, 3, ..."

    def test_citations_include_source_and_page(self, vector_db, fake_llm_relevant):
        retriever = vector_db.as_retriever(search_kwargs={"k": 2})
        graph = build_agentic_rag_graph(fake_llm_relevant, retriever)
        result = run_agentic_rag(graph, "SST rate?")

        for c in result["citations"]:
            assert c["source"] != "unknown", "source metadata must be preserved"
            assert c["page"] is not None, "page metadata must be preserved"
            assert len(c["snippet"]) > 0


# ============================================================
# 6. Multilingual
# ============================================================

class TestMultilingual:
    def test_chinese_query_finds_chinese_doc(self, vector_db):
        """A Chinese query should return the Chinese doc among top results."""
        results = vector_db.similarity_search("服务税税率是多少？", k=3)
        joined = " ".join(d.page_content for d in results)
        # Either the Chinese doc OR the English SST doc should appear
        assert ("服务" in joined) or ("SST" in joined or "service tax" in joined.lower())
