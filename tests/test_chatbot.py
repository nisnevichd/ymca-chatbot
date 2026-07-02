# tests/test_chatbot.py
import pytest
from pathlib import Path

from chatbot import (
    _classify_retrieval_scope,
    _read_scope_specific_sources,
    answer_question,
    DEFAULT_DOCS_DIR,
    ROOT_DIR,
)


# ---------- Scope classification ----------

@pytest.mark.parametrize("question,expected_scope", [
    ("What time does the branch open on Saturday?", "hours"),
    ("Are you open on holidays?", "hours"),
    ("What swim lessons do you offer?", "web"),
    ("How do I sign up for membership?", "web"),
    ("What is the whistleblower policy?", "handbook"),
    ("How much PTO do I get?", "handbook"),
    ("What is the meaning of life?", "all"),
    ("Are the basketball courts open?", "hours"),
    ("What time does the pool close?", "hours"),
    ("How do I sign up for basketball league?", "all"),
    ("When can I play pickleball?", "all"),  # nothing matches -> falls through
])
def test_scope_classification(question, expected_scope):
    assert _classify_retrieval_scope(question) == expected_scope


# ---------- hours.json resolution (regression test for the bug we just fixed) ----------

def test_hours_file_exists_at_root():
    hours_path = ROOT_DIR / "hours.json"
    assert hours_path.exists(), "hours.json should live at project root, not inside docs/"

def test_hours_scope_returns_content():
    context = _read_scope_specific_sources(DEFAULT_DOCS_DIR, "What are your hours today?")
    assert context.strip() != "", "hours scope should return non-empty content"


# ---------- End-to-end answer quality ----------
# These hit the real index + real Claude API — mark them so you can skip when offline/no API key

@pytest.mark.integration
class TestRealAnswers:

    def test_hours_question(self):
        answer = answer_question("What time does the Porter Ranch branch open?")
        assert "not enough information" not in answer.lower()
        assert "could not find" not in answer.lower()

    def test_handbook_policy_question(self):
        answer = answer_question("What is the YMCA's policy on employee benefits?")
        assert "not enough information" not in answer.lower()
        assert "could not find" not in answer.lower()

    def test_web_program_question(self):
        answer = answer_question("What aquatics programs does the YMCA offer?")
        assert "not enough information" not in answer.lower()
        assert "could not find" not in answer.lower()

    def test_membership_question(self):
        answer = answer_question("How do I sign up for a YMCA membership?")
        assert "not enough information" not in answer.lower()


# ---------- Edge cases ----------

def test_empty_question():
    answer = answer_question("")
    assert "Please enter a question" in answer

def test_whitespace_only_question():
    answer = answer_question("   ")
    assert "Please enter a question" in answer

def test_nonsense_question():
    # Should not crash, should gracefully say it doesn't have info
    answer = answer_question("asdkfjaskldfjaslkdfj random gibberish xyz123")
    assert isinstance(answer, str)
    assert len(answer) > 0