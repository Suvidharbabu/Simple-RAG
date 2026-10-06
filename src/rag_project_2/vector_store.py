import hashlib
from pathlib import Path
from typing import List, Optional

import chromadb
import numpy as np
from langchain_core.documents import Document

from . import config

COLLECTION_METADATA = {"description": "PDF chunks for RAG project", "hnsw:space": "cosine"}


class VectorStore:
    """ChromaDB collection using cosine distance (similarity = 1 - distance).

    Pass persist_directory=None for an in-memory store (used when tuning chunk sizes)."""

    def __init__(self, collection_name: str = config.COLLECTION_NAME,
                 persist_directory: Optional[Path] = config.VECTOR_STORE_DIR):
        self.collection_name = collection_name
        if persist_directory is None:
            self.client = chromadb.EphemeralClient()
        else:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata=COLLECTION_METADATA)

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection (needed after changing chunk settings, or old chunks linger)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name, metadata=COLLECTION_METADATA)

    def add_documents(self, documents: List[Document], embeddings: np.ndarray, batch_size: int = 1000):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")

        ids, metadatas, texts = [], [], []
        for doc in documents:
            # Deterministic ID, so re-ingesting the same chunks updates instead of duplicating
            key = f"{doc.metadata.get('source', '')}|{doc.metadata.get('page', '')}|{doc.page_content}"
            ids.append(f"doc_{hashlib.md5(key.encode('utf-8')).hexdigest()}")
            metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            texts.append(doc.page_content)

        # Chroma limits the size of a single request, so upsert in batches
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                metadatas=metadatas[start:end],
                documents=texts[start:end],
                embeddings=[e.tolist() for e in embeddings[start:end]],
            )
        print(f"Upserted {len(ids)} chunks. Collection now has {self.count()} chunks.")
