import os
from time import time
import streamlit as st

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Hybrid search + reranking
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


DB_DOCS_LIMIT = 10


def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk.content
   # st.session_state.messages.append({"role": "assistant", "content": response_message})


def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "" # Remove the RAG Response prefix
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk
   # st.session_state.messages.append({"role": "assistant", "content": response_message})


def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(doc_file.read())
                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Unsupported document type: {doc_file.type}")
                            continue
                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)
                    except Exception as e:
                        st.error(f"Failed to load {doc_file.name}: {e}")
                    finally:
                        os.remove(file_path)
                else:
                    st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT})")
        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document(s) loaded: {', '.join([doc_file.name for doc_file in st.session_state.rag_docs])}", icon="✅")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                try:
                    # Set a UA and timeout — some sites hang or 403 without them.
                    loader = WebBaseLoader(
                        url,
                        requests_kwargs={"timeout": 15},
                        header_template={
                            "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"
                        },
                    )
                    docs = loader.load()
                    st.session_state.rag_sources.append(url)
                    _split_and_load_docs(docs)
                    st.toast(f"URL loaded successfully: {url}", icon="✅")
                except Exception as e:
                    st.error(f"Failed to load from URL: {e}")
            else:
                st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT})")


@st.cache_resource(show_spinner="Loading embedding model (multilingual-e5-base)...")
def get_local_embedding_model():
    """Cached: only load the ~1.1GB embedding model once per session."""
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        # Batch more docs per forward pass -> big speedup on CPU
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    )


def initialize_vector_db(docs):
    embedding = get_local_embedding_model()
    vector_db = FAISS.from_documents(documents=docs, embedding=embedding)
    return vector_db


def _split_and_load_docs(docs):
    # chunk_size=5000 was WAY too big — e5-base truncates at 512 tokens,
    # so most of each chunk was silently ignored. 800/120 is the sweet spot
    # for e5-family models and gives much better retrieval quality too.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(chunks)
    else:
        st.session_state.vector_db.add_documents(chunks)

    # Also track raw chunks so BM25 (keyword search) can index them.
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []
    st.session_state.all_chunks.extend(chunks)

    # Invalidate cached BM25 / hybrid retriever — new docs came in.
    st.session_state.pop("_bm25_retriever", None)
    st.session_state.pop("_hybrid_retriever", None)


def get_retriever(vector_db, k: int = 4):
    """Plain dense retriever (FAISS only) — used as fallback."""
    return vector_db.as_retriever(search_kwargs={"k": k})


# --- Hybrid search + Reranking (optimized retrieval) ---

@st.cache_resource(show_spinner="Loading reranker (BAAI/bge-reranker-base)...")
def _load_reranker():
    """Cached: only download & load the ~278MB cross-encoder once per session."""
    return HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")


def get_hybrid_retriever(vector_db, chunks, k_dense: int = 6, k_sparse: int = 6):
    """
    Hybrid = BM25 (keyword) + FAISS (semantic) via reciprocal rank fusion.
    - BM25 catches exact keyword/entity matches (numbers, proper nouns, code)
    - FAISS catches semantic paraphrases
    Weights favor semantic slightly (0.6 vs 0.4) but both contribute.

    BM25 is cached in session_state and only rebuilt when new docs are added
    (see `_split_and_load_docs` which invalidates the cache).
    """
    bm25 = st.session_state.get("_bm25_retriever")
    if bm25 is None:
        bm25 = BM25Retriever.from_documents(chunks)
        st.session_state._bm25_retriever = bm25
    bm25.k = k_sparse
    dense = vector_db.as_retriever(search_kwargs={"k": k_dense})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])


def get_reranked_retriever(base_retriever, top_n: int = 4):
    """
    Wraps any retriever with a cross-encoder rerank pass.
    Cross-encoder scores (query, doc) as a pair -> much more accurate than
    bi-encoder cosine similarity, at the cost of latency (~50-200ms for top-12).
    """
    reranker = CrossEncoderReranker(model=_load_reranker(), top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )


def get_optimized_retriever(vector_db, chunks, top_n: int = 4):
    """Hybrid search -> cross-encoder rerank. The full recipe."""
    hybrid = get_hybrid_retriever(vector_db, chunks, k_dense=6, k_sparse=6)
    return get_reranked_retriever(hybrid, top_n=top_n)


def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to retrieve relevant context."),
    ])
    return create_history_aware_retriever(llm, retriever, prompt)


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use only the provided context to answer the question directly and concisely. Do not explain your reasoning. Do not include your thinking process. Please exclude your thinking process\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)
