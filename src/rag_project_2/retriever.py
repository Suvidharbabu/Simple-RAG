from typing import Any, Dict, List

from . import config
from .embeddings import EmbeddingManager
from .vector_store import VectorStore


class RAGRetriever:
    """Embeds the query and returns the most similar chunks above a score threshold."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = config.TOP_K, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

        if not results["documents"] or not results["documents"][0]:
            return []

        retrieved = []
        rows = zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0])
        for rank, (doc_id, content, metadata, distance) in enumerate(rows, start=1):
            similarity = 1 - distance
            if similarity >= score_threshold:
                retrieved.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "similarity_score": similarity,
                    "distance": distance,
                    "rank": rank,
                })
        return retrieved
