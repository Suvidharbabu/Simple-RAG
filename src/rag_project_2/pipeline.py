"""RAG pipeline: answer from the PDFs, fall back to an LLM with live web search when they don't cover the question."""
import re
from typing import Any, Dict, List

from langchain_groq import ChatGroq

from . import config
from .retriever import RAGRetriever

NOT_FOUND_TOKEN = "NOT_FOUND_IN_CONTEXT"


def get_llm(model: str = config.LLM_MODEL) -> ChatGroq:
    return ChatGroq(groq_api_key=config.groq_api_key(), model_name=model, temperature=0.1, max_tokens=1024)


def get_web_search_llm() -> ChatGroq:
    """Groq's built-in browser_search tool is only supported by openai/gpt-oss-20b and gpt-oss-120b."""
    return get_llm(config.WEB_SEARCH_MODEL)


def get_model_name(llm) -> str:
    return getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown model"


def clean_answer(text: str) -> str:
    """Remove reasoning blocks (<think>...</think>) and browser_search citation markers like 【0†L707-L724】."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"【[^】]*】", "", text)
    return text.strip()


def web_sources(web_llm) -> List[Dict[str, Any]]:
    return [{"source": f"LLM + web search: {get_model_name(web_llm)}", "page": None, "score": None,
             "preview": "Answered from a live web search"}]


def llm_fallback(query: str, web_llm) -> str:
    """Answer with live web search. Raises on API errors; rag_advanced handles them."""
    prompt = f"""The user's documents did not contain the answer to this question.
Search the web for up-to-date information and answer concisely. If you can't find it, say so.

Question: {query}

Answer:"""
    searching_llm = web_llm.bind(tools=[{"type": "browser_search"}], tool_choice="required")
    return clean_answer(searching_llm.invoke(prompt).content)


def _fallback_result(query: str, web_llm, confidence: float) -> Dict[str, Any]:
    try:
        return {"answer": llm_fallback(query, web_llm), "source": web_sources(web_llm),
                "confidence": confidence, "answered_from": "llm"}
    except Exception as e:
        return {"answer": f"Web search fallback failed: {e}", "source": [],
                "confidence": confidence, "answered_from": "error"}


def rag_advanced(query: str, retriever: RAGRetriever, llm, web_llm=None, top_k: int = config.TOP_K,
                 min_score: float = config.MIN_SCORE, return_context: bool = False,
                 fallback: bool = True) -> Dict[str, Any]:
    """Returns {'answer', 'source', 'confidence', 'answered_from', ['context']}.

    answered_from is 'pdf', 'llm' (web search fallback), 'none' (not found, fallback disabled) or 'error'.
    Page numbers in 'source' are 1-based."""
    web_llm = web_llm or get_web_search_llm()
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        if not fallback:
            output = {"answer": "No relevant context found.", "source": [], "confidence": 0.0, "answered_from": "none"}
        else:
            output = _fallback_result(query, web_llm, 0.0)
        if return_context:
            output["context"] = ""
        return output

    context = "\n\n".join(doc["content"] for doc in results)
    sources = [{
        "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
        "page": doc["metadata"]["page"] + 1 if isinstance(doc["metadata"].get("page"), int) else None,
        "score": doc["similarity_score"],
        "preview": doc["content"][:300] + "...",
    } for doc in results]
    confidence = max(doc["similarity_score"] for doc in results)

    prompt = f"""Use the following context to answer the question concisely.
If the context does not contain the answer, reply with exactly: {NOT_FOUND_TOKEN}

Context:
{context}

Question: {query}

Answer:"""
    try:
        answer_text = clean_answer(llm.invoke(prompt).content)
    except Exception as e:
        output = {"answer": f"LLM call failed: {e}", "source": sources, "confidence": confidence,
                  "answered_from": "error"}
    else:
        if NOT_FOUND_TOKEN not in answer_text:
            output = {"answer": answer_text, "source": sources, "confidence": confidence, "answered_from": "pdf"}
        elif fallback:
            output = _fallback_result(query, web_llm, confidence)
        else:
            output = {"answer": "The answer was not found in the documents.", "source": [],
                      "confidence": confidence, "answered_from": "none"}

    if return_context:
        output["context"] = context
    return output
