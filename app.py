"""Streamlit UI.  Run with:  streamlit run app.py"""
import streamlit as st

from rag_project_2 import build_retriever, config
from rag_project_2.pipeline import get_llm, get_web_search_llm, rag_advanced

st.set_page_config(page_title="PDF Q&A", page_icon="📄")


@st.cache_resource
def load_components():
    return build_retriever(), get_llm(), get_web_search_llm()


retriever, llm, web_llm = load_components()

st.title("Ask your PDFs")
st.caption(f"{retriever.vector_store.count()} chunks indexed · answers by {config.LLM_MODEL}, "
           f"web fallback by {config.WEB_SEARCH_MODEL}")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, config.TOP_K)
    min_score = st.slider("Minimum similarity", 0.0, 1.0, config.MIN_SCORE, 0.05)
    fallback = st.toggle("Fall back to web search", value=True)

if retriever.vector_store.count() == 0:
    st.warning("The vector store is empty. Run `rag-project-2 ingest` first.")
    st.stop()

query = st.text_input("Question", placeholder="e.g. What did Jon Snow name his direwolf?")
if query:
    with st.spinner("Thinking..."):
        result = rag_advanced(query, retriever, llm, web_llm, top_k=top_k, min_score=min_score, fallback=fallback)

    badge = {"pdf": "📄 From your PDFs", "llm": "🌐 From web search",
             "none": "❔ Not found in PDFs", "error": "⚠️ Error"}[result["answered_from"]]
    st.markdown(f"**{badge}** · confidence {result['confidence']:.2f}")
    if result["answered_from"] == "error":
        st.error(result["answer"])
    else:
        st.write(result["answer"])

    if result["source"]:
        st.subheader("Sources")
        for s in result["source"]:
            title = s["source"] + (f" — page {s['page']}" if s["page"] else "")
            if s["score"] is not None:
                title += f" (score {s['score']:.2f})"
            with st.expander(title):
                st.write(s["preview"])
