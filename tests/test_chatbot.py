import shutil
from pathlib import Path

from chatbot import answer_question, ingest_documents


def test_ingest_documents_builds_index_from_local_sources(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "policy.txt").write_text(
        "YMCA branch hours are 6 AM to 8 PM on weekdays.",
        encoding="utf-8",
    )

    persist_dir = tmp_path / "data"
    index, metadata = ingest_documents(
        docs_dir=docs_dir,
        persist_dir=persist_dir,
        collection_name="test_collection",
        include_web=False,
    )

    assert index is not None
    assert metadata["collection_name"] == "test_collection"
    assert metadata["document_count"] >= 1


def test_answer_question_returns_context_based_response(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "policy.txt").write_text(
        "Branch hours are 6 AM to 8 PM on weekdays.",
        encoding="utf-8",
    )

    persist_dir = tmp_path / "data"
    ingest_documents(
        docs_dir=docs_dir,
        persist_dir=persist_dir,
        collection_name="test_collection",
        include_web=False,
    )

    answer = answer_question(
        "What are branch hours?",
        persist_dir=persist_dir,
        collection_name="test_collection",
        api_key=None,
    )

    assert "6 am" in answer.lower()


def test_ingest_documents_includes_uppercase_pdf_files(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    repo_pdf = Path(__file__).resolve().parents[1] / "docs" / "Team Member Handbook.PDF"
    shutil.copy2(repo_pdf, docs_dir / "Team Member Handbook.PDF")

    persist_dir = tmp_path / "data"
    _, metadata = ingest_documents(
        docs_dir=docs_dir,
        persist_dir=persist_dir,
        collection_name="test_collection",
        include_web=False,
    )

    assert any("Team Member Handbook.PDF" in source for source in metadata["source_files"])
