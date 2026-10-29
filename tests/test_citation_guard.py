from qsr_assistant.agent.citation_guard import repair_citation


def test_missing_citation_is_repaired_for_exact_single_source_excerpt():
    answer = "The Large Fries costs $3.40."
    sources = [
        {
            "doc_id": "menu-large-fries",
            "title": "Large Fries",
            "content": "Large Fries - Price: The Large Fries costs $3.40.",
        }
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is True
    assert repaired == (
        "The Large Fries costs $3.40. [menu-large-fries]"
    )


def test_malformed_citation_is_normalized_only_for_known_id():
    answer = "The Large Fries costs $3.40 (doc_id: menu-large-fries)."
    sources = [
        {
            "doc_id": "menu-large-fries",
            "content": "The Large Fries costs $3.40.",
        }
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is True
    assert "[menu-large-fries]" in repaired
    assert "(doc_id:" not in repaired


def test_unknown_malformed_citation_is_not_repaired():
    answer = "The Large Fries costs $3.40 (doc_id: invented-doc)."
    sources = [
        {
            "doc_id": "menu-large-fries",
            "content": "The Large Fries costs $3.40.",
        }
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is False
    assert repaired == answer


def test_paraphrase_without_citation_is_not_repaired():
    answer = "A large portion of fries costs $3.40."
    sources = [
        {
            "doc_id": "menu-large-fries",
            "content": "The Large Fries costs $3.40.",
        }
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is False
    assert repaired == answer


def test_ambiguous_exact_excerpt_is_not_repaired():
    answer = "The Large Fries costs $3.40."
    sources = [
        {
            "doc_id": "menu-large-fries",
            "content": "The Large Fries costs $3.40.",
        },
        {
            "doc_id": "menu-special-fries",
            "content": "The Large Fries costs $3.40.",
        },
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is False
    assert repaired == answer


def test_multiple_sentences_without_citations_are_not_repaired():
    answer = "The Large Fries costs $3.40. They are available all day."
    sources = [
        {
            "doc_id": "menu-large-fries",
            "content": (
                "The Large Fries costs $3.40. "
                "They are available all day."
            ),
        }
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is False
    assert repaired == answer


def test_existing_invalid_square_bracket_citation_is_not_repaired():
    answer = "The Large Fries costs $3.40 [invented-doc]."
    sources = [
        {
            "doc_id": "menu-large-fries",
            "content": "The Large Fries costs $3.40.",
        }
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is False
    assert repaired == answer



def test_known_but_unrelated_malformed_citation_is_not_repaired():
    answer = "The account locks for 15 minutes (doc_id: faq-refunds)."
    sources = [
        {
            "doc_id": "faq-app-login",
            "content": "The account locks for 15 minutes after repeated wrong passwords.",
        },
        {
            "doc_id": "faq-refunds",
            "content": "Refunds are processed within 3 to 5 business days.",
        },
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is False
    assert repaired == answer


def test_malformed_citation_is_repaired_only_when_its_source_supports_answer():
    answer = "The Large Fries costs $3.40 (doc_id: menu-large-fries)."
    sources = [
        {
            "doc_id": "menu-large-fries",
            "content": "The Large Fries costs $3.40.",
        },
        {
            "doc_id": "faq-refunds",
            "content": "Refunds are processed within 3 to 5 business days.",
        },
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is True
    assert repaired == "The Large Fries costs $3.40 [menu-large-fries]."


def test_dangling_doc_id_is_not_repaired():
    answer = "Try again after 15 minutes, doc id:."
    sources = [
        {
            "doc_id": "faq-app-login",
            "content": "Try again after 15 minutes.",
        }
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is False
    assert repaired == answer


def test_answer_cannot_match_across_separate_evidence_fields():
    answer = "Large Fries costs $3.40."
    sources = [
        {
            "doc_id": "menu-large-fries",
            "title": "Large Fries costs",
            "content": "$3.40 is a separate field.",
        }
    ]

    repaired, changed = repair_citation(answer, sources)

    assert changed is False
    assert repaired == answer
