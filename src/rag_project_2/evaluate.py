"""Run the eval set in data/eval/questions.json against the ingested vector store.

    python -m rag_project_2.evaluate                   # retrieval + answers (calls Groq)
    python -m rag_project_2.evaluate --retrieval-only  # no LLM calls

Metrics:
  retrieval hit  - for questions the PDFs answer, a retrieved chunk from the right file contains the evidence phrase
  route correct  - answered_from matches expected_route ('pdf' or 'llm')
  answer correct - the answer contains one of answer_keywords (case-insensitive)
"""
import argparse
import json
from typing import Any, Dict, List

from . import build_retriever, config


def load_questions() -> List[Dict[str, Any]]:
    with open(config.EVAL_FILE, encoding="utf-8") as f:
        return json.load(f)


def retrieval_hit(item: Dict[str, Any], results: List[Dict[str, Any]]) -> bool:
    evidence = item["evidence"].lower()
    return any(r["metadata"].get("source_file") == item["source_file"] and evidence in r["content"].lower()
               for r in results)


def answer_correct(item: Dict[str, Any], answer: str) -> bool:
    return any(k.lower() in answer.lower() for k in item["answer_keywords"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--min-score", type=float, default=config.MIN_SCORE)
    args = parser.parse_args()

    questions = load_questions()
    retriever = build_retriever()
    if retriever.vector_store.count() == 0:
        raise SystemExit("Vector store is empty. Run `rag-project-2 ingest` first.")

    if not args.retrieval_only:
        from .pipeline import get_llm, get_web_search_llm, rag_advanced
        llm, web_llm = get_llm(), get_web_search_llm()

    retrieval_hits = route_hits = answer_hits = 0
    in_scope = [q for q in questions if q["expected_route"] == "pdf"]

    for item in questions:
        results = retriever.retrieve(item["question"], top_k=args.top_k, score_threshold=args.min_score)
        top = results[0]["similarity_score"] if results else 0.0
        line = f"[{item['expected_route']}] top={top:.2f}"

        if item["expected_route"] == "pdf":
            hit = retrieval_hit(item, results)
            retrieval_hits += hit
            line += f" retrieval={'HIT ' if hit else 'MISS'}"

        if not args.retrieval_only:
            result = rag_advanced(item["question"], retriever, llm, web_llm, top_k=args.top_k, min_score=args.min_score)
            route_ok = result["answered_from"] == item["expected_route"]
            ans_ok = answer_correct(item, result["answer"])
            route_hits += route_ok
            answer_hits += ans_ok
            line += f" route={result['answered_from']}{'' if route_ok else '(X)'} answer={'OK' if ans_ok else 'WRONG'}"

        print(f"{line} | {item['question']}")
        if not args.retrieval_only and not ans_ok:
            print(f"      -> {result['answer'][:200]!r}")

    print(f"\nRetrieval hit rate: {retrieval_hits}/{len(in_scope)} ({retrieval_hits / len(in_scope):.0%})")
    if not args.retrieval_only:
        print(f"Route accuracy:     {route_hits}/{len(questions)} ({route_hits / len(questions):.0%})")
        print(f"Answer accuracy:    {answer_hits}/{len(questions)} ({answer_hits / len(questions):.0%})")


if __name__ == "__main__":
    main()
