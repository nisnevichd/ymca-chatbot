import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

# Load environment variables from the project .env file before any API clients are created.
ROOT_DIR = Path(__file__).resolve().parent


def _load_environment() -> None:
    """Load the project .env file reliably from the repository root or current working directory."""
    env_candidates = [ROOT_DIR / ".env", Path.cwd() / ".env"]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)

    if not os.getenv("ANTHROPIC_API_KEY"):
        load_dotenv(dotenv_path=ROOT_DIR / ".env", override=True)


_load_environment()

DEFAULT_DOCS_DIR = ROOT_DIR / "docs"
DEFAULT_PERSIST_DIR = ROOT_DIR / "data"
DEFAULT_COLLECTION_NAME = "ymca_knowledge"
DEFAULT_WEB_URLS = [
    "https://www.ymcala.org/programs",
    "https://www.ymcala.org/aquatics",
    "https://www.ymcala.org/membership",
    "https://www.ymcala.org/handball/",
]
DEFAULT_WEB_URL = DEFAULT_WEB_URLS[0]

# Hard limits to prevent oversized prompts
MAX_CONTEXT_CHUNKS = 5
MAX_CHUNK_CHARS = 500
MAX_CONTEXT_CHARS = 4000


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


def _normalize_url(url: str, base_url: Optional[str] = None) -> str:
    """Normalize a URL and remove fragments while keeping only http/https links."""
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme:
        if not base_url:
            return ""
        parsed = urlparse(urljoin(base_url, url))
    if parsed.scheme not in {"http", "https"}:
        return ""
    if not parsed.netloc:
        return ""
    return urlunparse(parsed._replace(fragment=""))


def _is_within_domain(url: str, domain: str = "ymcala.org") -> bool:
    """Return True when a URL belongs to the YMCA domain."""
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    return host == domain or host.endswith(f".{domain}")


def _extract_main_text_from_html(html: str) -> str:
    """Extract clean body text from HTML, ignoring nav, headers, footers, and scripts."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "img", "button",
                     "input", "select", "option", "form", "iframe"]):
        tag.decompose()
    for tag_name in ["header", "nav", "footer", "aside"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    content_candidates = [
        soup.find("main"),
        soup.find("article"),
        soup.find(attrs={"role": "main"}),
        soup.select_one(".main-content"),
        soup.select_one(".content"),
        soup.select_one(".entry-content"),
        soup.select_one(".page-content"),
        soup.body,
    ]
    content_node = next((n for n in content_candidates if n is not None), None)
    if content_node is None:
        return ""
    text = content_node.get_text("\n", strip=True)
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned:
            continue
        if cleaned.lower() in {"skip to content", "menu", "close", "login", "join the y"}:
            continue
        lines.append(cleaned)
    return "\n".join(lines)


def _sanitize_filename(url: str) -> str:
    """Create a filesystem-safe filename from a URL."""
    parsed = urlparse(url)
    raw_path = parsed.path.rstrip("/") or "/"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", f"{parsed.netloc}{raw_path}").strip("_") or "home"
    return slug[:80]


def _crawl_ymca_site(start_url: str, output_dir: Path, max_pages: int = 200) -> list[Path]:
    """Recursively crawl the YMCA site and save each reachable page as plain text."""
    output_dir.mkdir(parents=True, exist_ok=True)

    queue = [start_url]
    visited_urls = set()
    saved_files = []
    pages_crawled = 0

    while queue and pages_crawled < max_pages:
        current_url = queue.pop(0)
        normalized_url = _normalize_url(current_url)
        if not normalized_url or normalized_url in visited_urls or not _is_within_domain(normalized_url):
            continue

        visited_urls.add(normalized_url)
        try:
            response = requests.get(normalized_url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network-dependent path
            print(f"Warning: could not fetch {normalized_url}: {exc}")
            continue

        text = _extract_main_text_from_html(response.text)
        if text:
            filename = f"{_sanitize_filename(normalized_url)}_{pages_crawled + 1:03d}.txt"
            output_path = output_dir / filename
            output_path.write_text(f"Source URL: {normalized_url}\n\n{text}", encoding="utf-8")
            saved_files.append(output_path)
            pages_crawled += 1

        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a", href=True):
            candidate_url = _normalize_url(link.get("href"), base_url=normalized_url)
            if not candidate_url or not _is_within_domain(candidate_url):
                continue
            if candidate_url not in visited_urls and candidate_url not in queue:
                queue.append(candidate_url)

    return saved_files


def _scrape_high_value_pages(docs_dir: Path, urls: Optional[list] = None) -> list:
    """Scrape the requested YMCA pages and save clean text files to docs/."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    for path in docs_dir.glob("*.txt"):
        path.unlink(missing_ok=True)
    saved_files = []
    for url in urls or DEFAULT_WEB_URLS:
        slug = _sanitize_filename(url)
        output_path = docs_dir / f"{slug}.txt"
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
        except Exception as exc:
            print(f"Warning: could not fetch {url}: {exc}")
            continue
        text = _extract_main_text_from_html(response.text)
        if text and len(text) > 40:
            output_path.write_text(f"Source URL: {url}\n\n{text}", encoding="utf-8")
            saved_files.append(output_path)
            print(f"Saved {url} -> {output_path}")
        else:
            print(f"Warning: no readable content found for {url}")
    return saved_files


def _load_documents(docs_dir: Path, include_web: bool, web_url: str) -> list:
    """Load the local handbook PDF and the scraped website content."""
    if include_web:
        try:
            scraped_files = _scrape_high_value_pages(docs_dir, urls=DEFAULT_WEB_URLS)
            print(f"Scraped {len(scraped_files)} YMCA website page(s) into {docs_dir}")
        except Exception as exc:
            print(f"Warning: could not scrape the YMCA website: {exc}")
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
    allowed_extensions = {".pdf", ".txt", ".md", ".html"}
    input_files = [
        path for path in sorted(docs_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]
    if not input_files:
        raise ValueError(f"No supported source documents were found in {docs_dir}")
    reader = SimpleDirectoryReader(input_files=[str(p) for p in input_files])
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
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
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
    chunk_nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
    )
    metadata = {
        "collection_name": collection_name,
        "document_count": len(documents),
        "chunk_count": len(chunk_nodes),
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


def _classify_retrieval_scope(question: str) -> str:
    """Classify whether a question should use handbook content, public web content, or branch-hours data."""
    question_lower = question.lower()

    if any(keyword in question_lower for keyword in ["hour", "hours", "branch hours", "open", "operating"]):
        return "hours"

    public_topics = [
        "aquatics",
        "swim",
        "program",
        "programs",
        "membership",
        "classes",
        "schedule",
        "youth",
        "camp",
        "sports",
        "handball",
        "wellness",
        "core values",
        "values",
    ]
    handbook_topics = [
        "whistleblower",
        "payroll",
        "leave",
        "arbitration",
        "discipline",
        "employee",
        "staff",
        "policy",
        "hr",
        "benefit",
        "benefits",
        "team member",
        "handbook",
    ]

    if any(keyword in question_lower for keyword in public_topics):
        return "web"
    if any(keyword in question_lower for keyword in handbook_topics):
        return "handbook"
    return "all"


def _truncate_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    """Truncate a passage to keep context compact."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + "..."


def _read_scope_specific_sources(docs_dir: Path, question: str) -> str:
    """Read the most relevant source files for employee, public-program, and branch-hours questions."""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return ""

    scope = _classify_retrieval_scope(question)
    if scope == "handbook":
        candidates = [path for path in docs_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"]
    elif scope == "web":
        candidates = [
            path
            for path in docs_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt" and "ymcala_org" in path.name.lower()
        ]
    elif scope == "hours":
        hours_path = docs_dir / "hours.json"
        if hours_path.exists():
            return hours_path.read_text(encoding="utf-8", errors="ignore")
        candidates = []
    else:
        candidates = [path for path in docs_dir.iterdir() if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}]

    parts = []
    for path in sorted(candidates):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if cleaned:
            parts.append(f"Source: {path.name}\n{cleaned}")

    return "\n\n".join(parts)


def _build_context_for_question(docs_dir: Path, question: str) -> str:
    """Build answer context from scope-specific sources before falling back to vector search."""
    return _read_scope_specific_sources(docs_dir, question)


def _build_context_from_nodes(nodes: list, max_chunks: int = MAX_CONTEXT_CHUNKS) -> str:
    """Convert retrieved nodes into a compact context window for Claude.
    
    Hard capped at MAX_CONTEXT_CHUNKS chunks and MAX_CONTEXT_CHARS total characters
    to prevent oversized prompts.
    """
    parts = []
    for i, node in enumerate(nodes[:max_chunks], start=1):
        text = node.get_content() if hasattr(node, "get_content") else str(node)
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            continue
        parts.append(f"Chunk {i}: {_truncate_text(cleaned)}")
    context = "\n\n".join(parts)
    # Final hard cap on total context size
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS].rstrip() + "..."
    return context


def _collect_relevant_nodes(index: VectorStoreIndex, question: str, scope: str) -> list:
    """Retrieve the most relevant nodes for a question using scope-aware queries."""
    if scope == "handbook":
        query = f"{question} handbook employee staff policy team member"
    elif scope == "web":
        query = f"{question} programs aquatics membership swim lessons classes core values"
    elif scope == "hours":
        query = f"{question} branch hours opening hours schedule"
    else:
        query = question
    # Only retrieve top 5 to keep context small
    retriever = index.as_retriever(similarity_top_k=5)
    return retriever.retrieve(query)


def _fallback_answer(question: str, context: str) -> str:
    """Return a grounded answer when no Anthropic API key is available."""
    context_snippet = context.strip()
    if not context_snippet:
        return "I could not find enough information in the uploaded handbook sources to answer that question."
    if question.lower().startswith(("what are", "what is")):
        return f"Based on the current sources: {context_snippet[:1200]}"
    return f"Based on the available sources: {context_snippet[:1500]}"


def answer_question(
    question: str,
    persist_dir: Optional[Path] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    api_key: Optional[str] = None,
) -> str:
    """Answer a question using indexed handbook content.
    
    Uses RAG: retrieves relevant chunks from ChromaDB, then sends only
    those chunks (capped at 4000 chars total) to Claude for answering.
    """
    if not question or not question.strip():
        return "Please enter a question about YMCA policies, branch hours, or handbook information."

    # Step 1: Load the vector index from ChromaDB
    index = load_index(persist_dir=persist_dir, collection_name=collection_name)

    # Step 2: Classify the question and retrieve relevant chunks
    scope = _classify_retrieval_scope(question)
    nodes = _collect_relevant_nodes(index, question, scope)

    # Step 3: Build compact context (hard capped to prevent token overflow)
    context = _build_context_from_nodes(nodes)

    if not context:
        scope_context = _read_scope_specific_sources(DEFAULT_DOCS_DIR, question)
        if scope_context:
            context = scope_context
        else:
            return "I could not find relevant information in the provided sources."

    # Step 4: Send question + context to Claude
    _load_environment()
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        prompt = (
            "You are a YMCA staff assistant. Answer using ONLY the context below. "
            "If the answer is not in the context, say you do not have enough information.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return _fallback_answer(question, context)