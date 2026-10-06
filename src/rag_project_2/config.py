"""Project-wide settings. Paths are resolved from the project root, so code works from any working directory."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdf_files"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
EVAL_FILE = DATA_DIR / "eval" / "questions.json"

# Separate from the notebook's whole-page "pdf_documents" collection
COLLECTION_NAME = "pdf_chunks"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking (chosen with `python -m rag_project_2.tune`)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 5
MIN_SCORE = 0.3

# LLMs (Groq). browser_search only works on the gpt-oss models.
LLM_MODEL = "qwen/qwen3.8-27b"
WEB_SEARCH_MODEL = "openai/gpt-oss-20b"


def groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(f"GROQ_API_KEY is not set. Add it to {PROJECT_ROOT / '.env'}")
    return key
