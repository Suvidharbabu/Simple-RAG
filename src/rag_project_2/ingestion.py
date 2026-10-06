"""Load PDFs and split them into overlapping chunks."""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def load_pdfs(pdf_dir: Path = config.PDF_DIR) -> List[Document]:
    """Load every PDF in the directory, one Document per page."""
    loader = DirectoryLoader(str(pdf_dir), glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=False)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_file"] = Path(doc.metadata.get("source", "unknown")).name
    print(f"Loaded {len(documents)} pages from {pdf_dir}")
    return documents


def split_documents(documents: List[Document], chunk_size: int = config.CHUNK_SIZE,
                    chunk_overlap: int = config.CHUNK_OVERLAP) -> List[Document]:
    """Split pages into chunks so each embedding covers one focused passage instead of a whole page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [chunk for chunk in splitter.split_documents(documents) if chunk.page_content.strip()]
    print(f"Split {len(documents)} pages into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
