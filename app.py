from __future__ import annotations

import textwrap
from typing import Any

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(
    page_title="MediRAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context.
Context: {context}
Question: {question}
Start the answer directly. No small talk please.
"""

LOCAL_MODEL_PATH = "google/flan-t5-base"
DB_FAISS_PATH = "vector_store/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&family=Space+Grotesk:wght@500;700&display=swap');

        :root {
            --bg-1: #f8fafc;
            --bg-2: #eef6ff;
            --ink: #0f172a;
            --muted: #475569;
            --brand: #0ea5e9;
            --brand-2: #14b8a6;
            --card: rgba(255, 255, 255, 0.82);
            --stroke: rgba(14, 165, 233, 0.22);
        }

        .stApp {
            font-family: 'Outfit', sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at 8% 12%, rgba(20, 184, 166, 0.18), transparent 28%),
                radial-gradient(circle at 92% 9%, rgba(14, 165, 233, 0.16), transparent 33%),
                linear-gradient(140deg, var(--bg-1), var(--bg-2));
        }

        .main .block-container {
            padding-top: 1.1rem;
            max-width: 1180px;
        }

        .hero {
            background: linear-gradient(120deg, rgba(14, 165, 233, 0.16), rgba(20, 184, 166, 0.18));
            border: 1px solid var(--stroke);
            border-radius: 24px;
            padding: 1.3rem 1.4rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(8px);
            animation: riseIn 420ms ease-out;
        }

        .hero h1 {
            margin: 0;
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.35rem, 3.3vw, 2rem);
            letter-spacing: 0.2px;
        }

        .hero p {
            margin: 0.45rem 0 0;
            color: var(--muted);
            font-size: 1rem;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 16px;
            background: var(--card);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            animation: riseIn 280ms ease-out;
        }

        [data-testid="stChatInput"] {
            position: sticky;
            bottom: 0;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.84), rgba(243, 250, 255, 0.92));
            border-right: 1px solid rgba(148, 163, 184, 0.3);
        }

        .metric-card {
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 14px;
            padding: 0.75rem 0.85rem;
            background: rgba(255, 255, 255, 0.75);
            margin-bottom: 0.6rem;
        }

        .metric-card b {
            color: var(--ink);
        }

        @keyframes riseIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 820px) {
            .main .block-container { padding-top: 0.7rem; }
            .hero { padding: 1rem; border-radius: 16px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading FAISS vector store...")
def get_vector_store() -> FAISS:
    embeddings = get_embedding_model()
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner="Loading local FLAN-T5 model...")
def get_llm() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH)
    return tokenizer, model


def set_custom_prompt(template: str) -> PromptTemplate:
    return PromptTemplate(template=template, input_variables=["context", "question"])


def format_docs(docs: list[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def rag_pipeline(question: str, top_k: int) -> dict[str, Any]:
    tokenizer, model = get_llm()
    prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

    db = get_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    docs = retriever.invoke(question)
    context = format_docs(docs)

    final_prompt = prompt.format(context=context, question=question)
    inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "result": answer,
        "source_documents": docs,
    }


def render_sidebar() -> int:
    with st.sidebar:
        st.markdown("### System Status")
        st.markdown('<div class="metric-card"><b>Model</b><br>google/flan-t5-base</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><b>Embeddings</b><br>all-MiniLM-L6-v2</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><b>Vector DB</b><br>FAISS (local)</div>', unsafe_allow_html=True)
        st.divider()
        top_k = st.slider("Retrieved chunks (k)", min_value=1, max_value=6, value=3)
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return top_k


def render_header() -> None:
    st.markdown(
        """
        <section class="hero">
            <h1>MediRAG Assistant</h1>
            <p>
                Ask medical questions grounded in your indexed clinical PDFs.
                Responses are generated from retrieved context to reduce hallucinations.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    top_k = render_sidebar()
    render_header()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for idx, source in enumerate(message["sources"], start=1):
                        preview = textwrap.shorten(source, width=450, placeholder="...")
                        st.markdown(f"{idx}. {preview}")

    prompt = st.chat_input("Ask a question from your medical knowledge base...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                response = rag_pipeline(prompt, top_k=top_k)
                answer = response["result"].strip() or "I don't know based on the available context."
                sources = [doc.page_content for doc in response["source_documents"]]
            except Exception as exc:
                answer = (
                    "I could not generate a response right now. "
                    "Please verify the FAISS index and local model are available.\n\n"
                    f"Error details: {exc}"
                )
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for idx, source in enumerate(sources, start=1):
                    preview = textwrap.shorten(source, width=450, placeholder="...")
                    st.markdown(f"{idx}. {preview}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    main()
