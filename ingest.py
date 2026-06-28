from pathlib import Path

from chatbot import DEFAULT_COLLECTION_NAME, DEFAULT_DOCS_DIR, DEFAULT_PERSIST_DIR, ingest_documents


if __name__ == "__main__":
    print("Loading handbook sources and rebuilding the knowledge base...")
    index, metadata = ingest_documents(
        docs_dir=Path(DEFAULT_DOCS_DIR),
        persist_dir=Path(DEFAULT_PERSIST_DIR),
        collection_name=DEFAULT_COLLECTION_NAME,
        include_web=True,
    )
    print(f"Knowledge base rebuilt successfully with {metadata['document_count']} document(s) and {metadata['chunk_count']} chunk(s).")
    print(f"Collection: {metadata['collection_name']}")
    print(f"Stored in: {metadata['persist_dir']}")