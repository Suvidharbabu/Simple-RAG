"""Compare chunk settings and look at similarity scores to choose CHUNK_SIZE / CHUNK_OVERLAP / MIN_SCORE.

    python -m rag_project_2.tune

Builds a throwaway in-memory index per chunk setting (the persistent store is not touched), then reports
the retrieval hit rate on the eval set and the top similarity score for each question."""
import statistics

from . import config
from .embeddings import EmbeddingManager
from .evaluate import load_questions, retrieval_hit
from .ingestion import load_pdfs, split_documents
from .retriever import RAGRetriever
from .vector_store import VectorStore

# (chunk_size, chunk_overlap); None = whole pages, like the original notebook
SETTINGS = [None, (500, 100), (800, 150), (1200, 200)]


def main():
    questions = load_questions()
    in_scope = [q for q in questions if q["expected_route"] == "pdf"]
    out_scope = [q for q in questions if q["expected_route"] != "pdf"]
    pages = load_pdfs()
    embedder = EmbeddingManager()

    for setting in SETTINGS:
        if setting is None:
            name, docs = "whole pages", pages
        else:
            name, docs = f"size={setting[0]} overlap={setting[1]}", split_documents(pages, *setting)
        store = VectorStore(collection_name=f"tune_{len(docs)}", persist_directory=None)
        store.add_documents(docs, embedder.generate_embeddings([d.page_content for d in docs], show_progress=True))
        retriever = RAGRetriever(store, embedder)

        hits, in_scores, out_scores = 0, [], []
        print(f"\n===== {name} ({len(docs)} docs)")
        for q in questions:
            results = retriever.retrieve(q["question"], top_k=config.TOP_K)
            top = results[0]["similarity_score"] if results else 0.0
            if q["expected_route"] == "pdf":
                hit = retrieval_hit(q, results)
                hits += hit
                in_scores.append(top)
                print(f"  {'HIT ' if hit else 'MISS'} top={top:.2f} | {q['question']}")
            else:
                out_scores.append(top)
                print(f"  out  top={top:.2f} | {q['question']}")

        print(f"  -> hit rate {hits}/{len(in_scope)}; in-scope top score min={min(in_scores):.2f} "
              f"median={statistics.median(in_scores):.2f}; out-of-scope top scores "
              f"{', '.join(f'{s:.2f}' for s in out_scores)}")
        store.client.delete_collection(store.collection_name)


if __name__ == "__main__":
    main()
