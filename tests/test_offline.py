from types import SimpleNamespace

from qsr_assistant.llm.offline import OfflineClient


def tool_message(hits, *, low_confidence=False):
    return SimpleNamespace(
        role="tool",
        name="search_knowledge",
        content="",
        data={"hits": hits, "low_confidence": low_confidence},
    )


def test_broad_burger_question_includes_all_retrieved_hits_and_citations():
    hits = [
        {"text": "Classic Burger costs $4.50.", "doc_id": "menu-classic-burger"},
        {"text": "Spicy Chicken Burger costs $5.40.", "doc_id": "menu-spicy-chicken-burger"},
        {"text": "Veggie Delight Burger costs $4.80.", "doc_id": "menu-veggie-delight-burger"},
    ]

    answer = OfflineClient._compose(
        [tool_message(hits)],
        query="What burgers are available and how much do they cost?",
    )

    for hit in hits:
        assert hit["text"] in answer
        assert f"[{hit['doc_id']}]" in answer


def test_specific_question_keeps_only_top_retrieval_hit():
    hits = [
        {"text": "Large Fries costs $3.40.", "doc_id": "menu-large-fries"},
        {"text": "Classic Burger costs $4.50.", "doc_id": "menu-classic-burger"},
    ]

    answer = OfflineClient._compose(
        [tool_message(hits)],
        query="How much do large fries cost?",
    )

    assert hits[0]["text"] in answer
    assert f"[{hits[0]['doc_id']}]" in answer
    assert hits[1]["text"] not in answer
    assert f"[{hits[1]['doc_id']}]" not in answer


def test_low_confidence_still_abstains_for_broad_question():
    hits = [
        {"text": "Unrelated result.", "doc_id": "unrelated-doc"},
        {"text": "Another unrelated result.", "doc_id": "another-doc"},
    ]

    answer = OfflineClient._compose(
        [tool_message(hits, low_confidence=True)],
        query="What burgers are available and how much do they cost?",
    )

    assert "Unrelated result." not in answer
    assert "[unrelated-doc]" not in answer
