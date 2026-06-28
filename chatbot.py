import os
from pathlib import Path
from typing import Optional, Tuple

import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = ROOT_DIR / "docs"
DEFAULT_PERSIST_DIR = ROOT_DIR / "data"
DEFAULT_COLLECTION_NAME = "ymca_knowledge"
DEFAULT_WEB_URL = "https://www.ymcala.org/handbook/"


def _configure_embedding_model() -> None:
    """Use a local embedding model when no OpenAI API key is available."""
    if os.getenv("OPENAI_API_KEY"):
        from llama_index.embeddings.openai import OpenAIEmbedding

        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        return

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception:
        Settings.embed_model = MockEmbedding(embed_dim=384)


_configure_embedding_model()


def _scrape_member_handbook(url: str, output_path: Path) -> None:
    """Fetch the public handbook page and store a plain-text copy for indexing."""
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    output_path.write_text(text, encoding="utf-8")


def _load_documents(docs_dir: Path, include_web: bool, web_url: str) -> list:
    """Load the local handbook PDF and the scraped website text into LlamaIndex documents."""
    if include_web:
        web_output_path = docs_dir / "member_handbook_web.txt"
        try:
            _scrape_member_handbook(web_url, web_output_path)
        except Exception as exc:  # pragma: no cover - network-dependent path
            print(f"Warning: could not scrape the handbook website: {exc}")

    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)

    allowed_extensions = {".pdf", ".txt", ".md", ".html"}
    input_files = [
        path
        for path in sorted(docs_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]

    if not input_files:
        raise ValueError(f"No supported source documents were found in {docs_dir}")

    reader = SimpleDirectoryReader(input_files=[str(path) for path in input_files])
    documents = reader.load_data()
    if not documents:
        raise ValueError(f"No documents were found in {docs_dir}")
    return documents


def ingest_documents(
    docs_dir: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    include_web: bool = True,
    web_url: str = DEFAULT_WEB_URL,
) -> Tuple[VectorStoreIndex, dict]:
    """Rebuild the Chroma-based knowledge base from the current source documents."""
    docs_dir = Path(docs_dir or DEFAULT_DOCS_DIR)
    persist_dir = Path(persist_dir or DEFAULT_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents = _load_documents(docs_dir=docs_dir, include_web=include_web, web_url=web_url)

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
    )

    metadata = {
        "collection_name": collection_name,
        "document_count": len(documents),
        "docs_dir": str(docs_dir),
        "persist_dir": str(persist_dir),
        "source_files": [doc.metadata.get("file_name", "unknown") for doc in documents],
    }
    return index, metadata


def load_index(
    persist_dir: Optional[Path] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> VectorStoreIndex:
    """Load an existing Chroma-backed index from disk."""
    persist_dir = Path(persist_dir or DEFAULT_PERSIST_DIR)
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)


def _fallback_answer(question: str, context: str) -> str:
    """Create a grounded answer when no Anthropic API key is available."""
    context_snippet = context.strip()
    if not context_snippet:
        return "I could not find enough information in the uploaded handbook sources to answer that question."

    if question.lower().startswith("what are") or question.lower().startswith("what is"):
        return f"Based on the current handbook sources, the relevant information says: {context_snippet[:1200]}"

    return f"Based on the available handbook sources: {context_snippet[:1500]}"


def answer_question(
    question: str,
    persist_dir: Optional[Path] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    api_key: Optional[str] = None,
) -> str:
    """Answer a question using the current indexed handbook content."""
    if not question or not question.strip():
        return "Please enter a question about YMCA policies, branch hours, or handbook information."

    index = load_index(persist_dir=persist_dir, collection_name=collection_name)
    retriever = index.as_retriever(similarity_top_k=8)
    nodes = retriever.retrieve(question)

    if not nodes:
        return "I could not find relevant information in the provided handbook sources."

    context_parts = [node.get_content() for node in nodes]
    context = "\n\n".join(context_parts)
    if len(context_parts) < 8:
        extra_nodes = index.as_retriever(similarity_top_k=12).retrieve(question)
        for extra_node in extra_nodes:
            if extra_node not in nodes:
                context_parts.append(extra_node.get_content())
        context = "\n\n".join(context_parts)

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        prompt = (
            "You are a YMCA staff assistant for internal YMCA staff use. Answer the user question using ONLY the source context below. "
            "Do not redact or omit sensitive internal information such as staff names, emails, phone numbers, contact details, or branch-specific guidance when it appears in the context. "
            "If the answer is not present in the context, say that you do not have enough information in the provided materials.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return _fallback_answer(question, context)
