"""PDF question answering with ChromaDB + Groq, falling back to live web search.

CLI (after `pip install -e .`):
    rag-project-2 ingest [--reset]
    rag-project-2 ask "Who gave Arya her sword?"
"""
import argparse

from . import config


def build_retriever():
    from .embeddings import EmbeddingManager
    from .retriever import RAGRetriever
    from .vector_store import VectorStore

    return RAGRetriever(VectorStore(), EmbeddingManager())


def ingest(reset: bool = False):
    from .embeddings import EmbeddingManager
    from .ingestion import load_pdfs, split_documents
    from .vector_store import VectorStore

    store = VectorStore()
    if reset:
        print(f"Resetting collection '{config.COLLECTION_NAME}'")
        store.reset()
    chunks = split_documents(load_pdfs())
    embeddings = EmbeddingManager().generate_embeddings([c.page_content for c in chunks], show_progress=True)
    store.add_documents(chunks, embeddings)


def ask(query: str):
    from .pipeline import get_llm, rag_advanced

    result = rag_advanced(query, build_retriever(), get_llm())
    print(f"\nAnswered from: {result['answered_from']} (confidence {result['confidence']:.2f})")
    print(f"Answer: {result['answer']}\n")
    for s in result["source"]:
        page = f" p.{s['page']}" if s["page"] else ""
        score = f" ({s['score']:.2f})" if s["score"] is not None else ""
        print(f"  - {s['source']}{page}{score}")


def main():
    parser = argparse.ArgumentParser(prog="rag-project-2")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="Chunk, embed and store the PDFs in data/pdf_files")
    p_ingest.add_argument("--reset", action="store_true", help="Drop existing chunks first")
    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("query")
    args = parser.parse_args()

    if args.command == "ingest":
        ingest(reset=args.reset)
    else:
        ask(args.query)
