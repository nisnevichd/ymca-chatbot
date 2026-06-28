import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from the local .env file before any Anthropic setup.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

from chatbot import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DOCS_DIR,
    DEFAULT_PERSIST_DIR,
    answer_question,
    ingest_documents,
)

st.set_page_config(page_title="YMCA Staff Assistant", page_icon="🏋️", layout="wide")


@st.cache_data(show_spinner=False)
def _run_ingestion() -> dict:
    """Rebuild the knowledge base and return metadata for display."""
    _, metadata = ingest_documents(
        docs_dir=DEFAULT_DOCS_DIR,
        persist_dir=DEFAULT_PERSIST_DIR,
        collection_name=DEFAULT_COLLECTION_NAME,
        include_web=True,
    )
    return metadata


st.title("YMCA Staff Assistant")
st.caption("Ask questions about handbook policies, branch hours, and staff guidance using the latest indexed documents.")

with st.sidebar:
    st.header("Knowledge base")
    st.write("This app uses the local handbook PDF and the public member handbook source.")
    if st.button("Refresh knowledge base", use_container_width=True):
        with st.spinner("Refreshing documents and rebuilding the index..."):
            metadata = _run_ingestion()
        st.success(f"Knowledge base refreshed with {metadata['document_count']} document(s).")

    st.markdown("---")
    st.caption("Update the source documents in the docs/ folder or replace the web source whenever policies change.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about the handbook or branch hours")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the handbook sources..."):
            answer = answer_question(
                question=prompt,
                persist_dir=DEFAULT_PERSIST_DIR,
                collection_name=DEFAULT_COLLECTION_NAME,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})