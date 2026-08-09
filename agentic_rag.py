"""
Agentic RAG pipeline built with LangGraph.

Flow:
    query -> [rewrite] -> [retrieve] -> [grade docs]
                                            |
                            (relevant) -----+----- (not relevant, retries left)
                                    |                    |
                                    v                    v
                             [generate answer]     [rewrite query] --> back to retrieve
                                    |
                                    v
                              answer + citations

Every step is an LLM decision point (or has one) — that's what makes it "agentic"
vs a fixed chain. The grader can send the query back for rewriting; the generator
is instructed to cite its sources.
"""

from typing import List, TypedDict, Literal
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END


# ------------------------- Graph State -------------------------

class AgenticRAGState(TypedDict):
    """Shared state passed between nodes."""
    original_question: str      # what the user actually asked
    question: str               # current (possibly rewritten) query
    documents: List[Document]   # retrieved chunks
    answer: str                 # final answer text
    citations: List[dict]       # [{source, page, snippet}, ...]
    retry_count: int            # number of rewrite retries used
    trace: List[str]            # human-readable log of what each node did


# ------------------------- Structured Outputs -------------------------

class GradeDocuments(BaseModel):
    """LLM's yes/no verdict on whether retrieved docs answer the question."""
    is_relevant: Literal["yes", "no"] = Field(
        description="Whether the retrieved documents are relevant enough to answer the question"
    )
    reason: str = Field(description="One-sentence justification")


# ------------------------- Nodes -------------------------

def make_rewrite_node(llm):
    """Reformulates the user's question into a better retrieval query."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a query optimizer for a vector-database retriever. "
         "Rewrite the user question into a concise, keyword-rich search query "
         "that will maximize semantic retrieval quality. "
         "Keep it in the same language as the question. "
         "Return ONLY the rewritten query, no explanation."),
        ("user", "Original question: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    def node(state: AgenticRAGState) -> dict:
        rewritten = chain.invoke({"question": state["question"]}).strip()
        # Strip surrounding quotes if the model added them
        rewritten = rewritten.strip('"\'')
        return {
            "question": rewritten,
            "trace": state["trace"] + [f"🔄 Rewrote query: '{rewritten}'"],
        }
    return node


def make_retrieve_node(retriever):
    """Pulls top-k chunks from the vector store."""
    def node(state: AgenticRAGState) -> dict:
        docs = retriever.invoke(state["question"])
        return {
            "documents": docs,
            "trace": state["trace"] + [f"📥 Retrieved {len(docs)} chunks"],
        }
    return node


def make_grade_node(llm):
    """LLM decides if retrieved docs are actually useful."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are grading whether retrieved documents contain enough information "
         "to answer the user's question. "
         "Answer 'yes' if at least one document is on-topic and useful, "
         "'no' if all documents are off-topic or irrelevant."),
        ("user",
         "Question: {question}\n\n"
         "Retrieved documents:\n{docs}\n\n"
         "Are these documents relevant enough to answer the question?"),
    ])
    grader = llm.with_structured_output(GradeDocuments)
    chain = prompt | grader

    def node(state: AgenticRAGState) -> dict:
        # Concatenate a preview of each doc so the grader has context
        docs_preview = "\n---\n".join(
            f"[Doc {i+1}] {d.page_content[:400]}"
            for i, d in enumerate(state["documents"])
        ) or "(no documents)"
        try:
            verdict = chain.invoke({
                "question": state["original_question"],
                "docs": docs_preview,
            })
            note = f"✅ Grader: {verdict.is_relevant} — {verdict.reason}"
        except Exception as e:
            # If grader fails (e.g. model doesn't support structured output well),
            # default to "yes" so we don't block the pipeline.
            verdict = GradeDocuments(is_relevant="yes", reason=f"grader-fallback: {e}")
            note = f"⚠️ Grader fallback (assuming relevant): {e}"
        return {"trace": state["trace"] + [note], "_grade": verdict.is_relevant}
    return node


def decide_after_grade(state: AgenticRAGState) -> str:
    """Conditional edge: retry rewrite, or move on to generation."""
    grade = state.get("_grade", "yes")
    if grade == "yes":
        return "generate"
    if state["retry_count"] >= 1:  # max 1 retry — keeps latency bounded
        return "generate"  # give up retrying, answer with what we have
    return "rewrite_retry"


def rewrite_retry_node(state: AgenticRAGState) -> dict:
    """Increments retry counter and resets question to original for a fresh rewrite."""
    return {
        "question": state["original_question"],  # rewrite from scratch, not from prior rewrite
        "retry_count": state["retry_count"] + 1,
        "trace": state["trace"] + [f"↩️ Retry #{state['retry_count']+1} — re-rewriting"],
    }


def make_generate_node(llm):
    """Produces the final answer with inline citation markers [1], [2], ..."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the user's question using ONLY the "
         "provided context. Cite sources inline using bracketed numbers like [1], [2] "
         "that correspond to the numbered documents below. "
         "If the context does not contain the answer, say so honestly — do not invent facts. "
         "Be concise. Do not include your reasoning process.\n\n"
         "Context:\n{context}"),
        ("user", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    def node(state: AgenticRAGState) -> dict:
        docs = state["documents"]
        context = "\n\n".join(
            f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
        ) or "(no context available)"
        answer = chain.invoke({
            "context": context,
            "question": state["original_question"],
        })
        citations = [
            {
                "index": i + 1,
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
                "snippet": d.page_content[:200].replace("\n", " "),
            }
            for i, d in enumerate(docs)
        ]
        return {
            "answer": answer,
            "citations": citations,
            "trace": state["trace"] + [f"✍️ Generated answer with {len(citations)} citations"],
        }
    return node


# ------------------------- Graph Builder -------------------------

def build_agentic_rag_graph(llm, retriever):
    """Compile the LangGraph state machine. Call once, reuse across queries."""
    graph = StateGraph(AgenticRAGState)

    graph.add_node("rewrite", make_rewrite_node(llm))
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("grade", make_grade_node(llm))
    graph.add_node("rewrite_retry", rewrite_retry_node)
    graph.add_node("generate", make_generate_node(llm))

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        decide_after_grade,
        {"generate": "generate", "rewrite_retry": "rewrite_retry"},
    )
    graph.add_edge("rewrite_retry", "rewrite")
    graph.add_edge("generate", END)

    return graph.compile()


def run_agentic_rag(graph, question: str) -> dict:
    """Convenience wrapper. Returns {answer, citations, trace}."""
    initial: AgenticRAGState = {
        "original_question": question,
        "question": question,
        "documents": [],
        "answer": "",
        "citations": [],
        "retry_count": 0,
        "trace": [],
    }
    final_state = graph.invoke(initial)
    return {
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "trace": final_state["trace"],
    }
