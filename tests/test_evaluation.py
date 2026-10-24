
from app.evaluate_answers import is_abstention
from app.evaluate import extract_text, is_relevant


def test_recognizes_missing_parental_leave_information():
    answer = "The evidence does not mention a parental leave duration."
    assert is_abstention(answer) is True


def test_recognizes_explicit_abstention():
    answer = "The document does not specify a parental leave policy."
    assert is_abstention(answer) is True


def test_does_not_classify_normal_answer_as_abstention():
    answer = "Employees must enable multi-factor authentication."
    assert is_abstention(answer) is False


def test_extracts_retrieved_chunk_text():
    result = {
        "text": "Employees must enable multi-factor authentication."
    }
    assert extract_text(result) == (
        "Employees must enable multi-factor authentication."
    )


def test_identifies_relevant_chunk():
    result = {
        "text": "Employees receive 20 days of paid leave per year."
    }
    anchor = "Employees receive 20 days of paid leave per year."
    assert is_relevant(result, anchor) is True


def test_unrelated_chunk_is_not_relevant():
    result = {
        "text": "Passwords must never be shared with colleagues."
    }
    anchor = "Employees receive 20 days of paid leave per year."
    assert is_relevant(result, anchor) is False


def test_empty_evidence_anchor_is_not_relevant():
    result = {
        "text": "Employees must enable multi-factor authentication."
    }
    assert is_relevant(result, "") is False
