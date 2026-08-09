"""
Shared pytest fixtures.

Design principles:
- No network calls, no API keys required to run tests.
- Fake LLM returns deterministic responses so we can assert on graph behavior.
- Tiny in-memory corpus so tests finish in seconds.
"""

import pytest
from typing import List
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


# --------------------------- Corpus ---------------------------

@pytest.fixture
def sample_docs() -> List[Document]:
    """Small multilingual test corpus with page metadata (mimics PyPDFLoader)."""
    return [
        Document(
            page_content="Malaysia's Sales and Service Tax (SST) is a consumption tax "
                         "levied on goods and services. The service tax rate is 6%.",
            metadata={"source": "sst_guide.pdf", "page": 0},
        ),
        Document(
            page_content="SST replaced the Goods and Services Tax (GST) in September 2018. "
                         "It is governed by the Service Tax Regulations 2018.",
            metadata={"source": "sst_guide.pdf", "page": 1},
        ),
        Document(
            page_content="The threshold for SST registration is RM500,000 per annum "
                         "for most service categories.",
            metadata={"source": "sst_guide.pdf", "page": 2},
        ),
        Document(
            page_content="马来西亚的销售与服务税（SST）是对商品和服务征收的消费税，"
                         "服务税税率为百分之六。",
            metadata={"source": "sst_zh.pdf", "page": 0},
        ),
        Document(
            page_content="Photosynthesis is the process by which plants convert sunlight "
                         "into chemical energy. This is unrelated to taxation.",
            metadata={"source": "biology.pdf", "page": 0},
        ),
    ]


# --------------------------- Fake embeddings ---------------------------

class HashEmbeddings(Embeddings):
    """
    Deterministic pseudo-embeddings based on token overlap.
    Not "smart" but stable enough to test wiring and retrieval logic
    without downloading a real model or hitting Cohere.
    """
    dim = 64
    vocab = [
        "sst", "tax", "service", "malaysia", "gst", "threshold", "rm500000",
        "sales", "consumption", "regulations", "2018", "6%", "500000",
        "photosynthesis", "plants", "sunlight",
        "销售", "服务", "税", "马来西亚", "消费"
    ]

    def _embed(self, text: str) -> List[float]:
        text_l = text.lower()
        vec = [0.0] * self.dim
        for i, word in enumerate(self.vocab):
            if word in text_l:
                vec[i % self.dim] += 1.0
        # avoid all-zero vectors (FAISS dislikes them)
        vec[-1] += 0.01
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


@pytest.fixture
def fake_embeddings():
    return HashEmbeddings()


@pytest.fixture
def vector_db(sample_docs, fake_embeddings):
    return FAISS.from_documents(sample_docs, fake_embeddings)


# --------------------------- Fake LLM ---------------------------

@pytest.fixture
def fake_llm_relevant():
    """
    Cycles through: [rewritten query, grader verdict JSON, final answer]
    Matches the order nodes call the LLM in Agentic RAG.
    """
    responses = [
        "What is Malaysia SST tax rate?",                       # rewrite
        '{"is_relevant": "yes", "reason": "on-topic"}',         # grade
        "The Malaysia SST service tax rate is 6% [1].",         # generate
    ]
    return FakeListChatModel(responses=responses)


@pytest.fixture
def fake_llm_irrelevant_then_relevant():
    """
    First round: grader says NOT relevant -> retry.
    Second round: grader says relevant -> generate answer.
    """
    responses = [
        "SST rate",                                              # rewrite #1
        '{"is_relevant": "no", "reason": "off-topic"}',          # grade  #1 -> retry
        "Malaysia service tax rate",                             # rewrite #2
        '{"is_relevant": "yes", "reason": "on-topic"}',          # grade  #2
        "The Malaysia SST service tax rate is 6% [1].",          # generate
    ]
    return FakeListChatModel(responses=responses)
