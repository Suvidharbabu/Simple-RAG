# RAG project 2: Ask your PDFs

Answers questions from the PDFs in `data/pdf_files/` using ChromaDB and a Groq LLM. When the PDFs don't contain the answer, it falls back to a live web search.

```
question ─► embed (all-MiniLM-L6-v2) ─► top-k chunks from ChromaDB (cosine)
              │ nothing above MIN_SCORE                │
              ▼                                        ▼
        web search (gpt-oss-20b          qwen answers from the chunks,
        + browser_search)       ◄──────  or replies NOT_FOUND_IN_CONTEXT
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -e .                 # installs the package, its dependencies and the `rag-project-2` command
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-key-here
```

Put your PDFs in `data/pdf_files/`.

## Usage

```bash
rag-project-2 ingest --reset           # chunk + embed the PDFs into data/vector_store (use --reset after changing chunk settings)
rag-project-2 ask "What did Jon Snow name his direwolf?"
streamlit run app.py                   # web UI
```

`answered_from` in each result shows where the answer came from:

| value   | meaning                                                      |
|---------|--------------------------------------------------------------|
| `pdf`   | answered from the retrieved chunks; sources list file + page |
| `llm`   | the PDFs didn't cover it, so it was answered by web search   |
| `none`  | not found and the fallback is disabled                       |
| `error` | a Groq API call failed; the answer contains the error        |

## Evaluation and tuning

`data/eval/questions.json` contains test questions. Each one lists where it should be answered from (`pdf` or `llm`), a phrase the retrieved chunk should contain, and keywords the answer should contain.

```bash
python -m rag_project_2.evaluate --retrieval-only   # retrieval hit rate only, no LLM calls
python -m rag_project_2.evaluate                    # also checks routing and answers (calls Groq)
python -m rag_project_2.tune                        # compares chunk sizes and prints similarity scores
```

The settings are in `src/rag_project_2/config.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_SCORE` and the model names.

## Layout

```
src/rag_project_2/
  config.py        paths, models, chunk/retrieval settings
  ingestion.py     load PDFs, split into chunks
  embeddings.py    sentence-transformers wrapper
  vector_store.py  ChromaDB collection
  retriever.py     query -> scored chunks
  pipeline.py      rag_advanced(): answer, sources, web fallback
  evaluate.py      eval-set runner
  tune.py          chunk-size / threshold comparison
app.py             Streamlit UI
notebook/          experiments
```

Notes:
- The `browser_search` tool only works on Groq's `openai/gpt-oss-20b` / `gpt-oss-120b` models, so the web fallback uses its own model. Other models reject it with `tools[0].type must be one of [function, mcp]`.
- PDFs and the vector store are git-ignored. Rebuild the store with `rag-project-2 ingest`.
