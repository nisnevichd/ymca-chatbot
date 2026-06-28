import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot import (
    _build_context_for_question,
    _classify_retrieval_scope,
    _crawl_ymca_site,
    _extract_main_text_from_html,
    _read_scope_specific_sources,
    answer_question,
    ingest_documents,
)


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


def test_extract_main_text_from_html_ignores_navigation_and_footer():
    html = """
    <html><body>
      <header>YMCA of Metropolitan Los Angeles</header>
      <nav>Programs Aquatics Youth</nav>
      <main>
        <h1>Membership Handbook</h1>
        <p>Welcome to the Y.</p>
      </main>
      <footer>Contact us</footer>
    </body></html>
    """

    text = _extract_main_text_from_html(html)

    assert "Membership Handbook" in text
    assert "Welcome to the Y." in text
    assert "Programs" not in text
    assert "Contact us" not in text


def test_classify_retrieval_scope_prioritizes_web_for_public_programs():
    assert _classify_retrieval_scope("What aquatics programs are available?") == "web"
    assert _classify_retrieval_scope("How do I file a whistleblower complaint?") == "handbook"
    assert _classify_retrieval_scope("Tell me about the YMCA") == "all"


def test_read_scope_specific_sources_uses_web_files_for_public_topics(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "www_ymcala_org_aquatics.txt").write_text(
        "Aquatics programs include swim lessons and water aerobics.",
        encoding="utf-8",
    )
    (docs_dir / "Team Member Handbook.PDF").write_bytes(b"employee handbook")

    content = _read_scope_specific_sources(docs_dir, "What aquatics programs are available?")

    assert "swim lessons" in content.lower()
    assert "employee handbook" not in content.lower()


def test_build_context_for_question_prefers_web_sources_for_public_topics(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "www_ymcala_org_aquatics.txt").write_text(
        "Aquatics programs include swim lessons and water aerobics.",
        encoding="utf-8",
    )
    (docs_dir / "Team Member Handbook.PDF").write_bytes(b"employee handbook")

    context = _build_context_for_question(docs_dir, "What aquatics programs are available?")

    assert "swim lessons" in context.lower()
    assert "employee handbook" not in context.lower()


def test_crawl_ymca_site_skips_duplicates_and_external_links(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    pages = {
        "https://www.ymcala.org/": "<html><body><a href='/programs'>Programs</a><a href='https://example.com/ignore'>Ignore</a></body></html>",
        "https://www.ymcala.org/programs": "<html><body><main><h1>Programs</h1><p>Programs page</p></main></body></html>",
        "https://www.ymcala.org/aquatics": "<html><body><main><h1>Aquatics</h1><p>Aquatics page</p></main></body></html>",
    }

    def fake_get(url, timeout, headers=None):
        return FakeResponse(pages[url])

    monkeypatch.setattr("chatbot.requests.get", fake_get)

    output_dir = tmp_path / "site_pages"
    files = _crawl_ymca_site("https://www.ymcala.org/", output_dir=output_dir, max_pages=5)

    assert len(files) == 2
    assert output_dir.exists()
