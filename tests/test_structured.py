import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from qsr_assistant.api.main import app
from qsr_assistant.llm import BaseClient, LLMResponse
from qsr_assistant.llm.structured import (
    StructuredError,
    Ticket,
    extract_ticket,
    generate_structured,
    parse_json_object,
    rule_extract_ticket,
)

GOOD = ('{"category":"food_quality","severity":"medium","store_id":3,"items":["large fries"],'
        '"summary":"Cold fries","needs_human":true}')


class Replies(BaseClient):
    name, online, model = "fake", True, "fake"

    def __init__(self, *replies):
        self.replies, self.n = list(replies), 0

    def chat(self, system, messages, tools=None, max_tokens=0):
        self.n += 1
        self.last = messages
        return LLMResponse(self.replies.pop(0) if len(self.replies) > 1 else self.replies[0])


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_repair_loop_feeds_the_error_back():
    c = Replies("{not json", '{"category":"nonsense"}', GOOD)
    t, repairs = generate_structured(c, Ticket, "x", "my fries were cold at store 3")
    assert t.category == "food_quality" and repairs == 2 and c.n == 3
    assert "did not validate" in c.last[-1].content


def test_gives_up_after_retries():
    with pytest.raises(StructuredError):
        generate_structured(Replies("{bad"), Ticket, "x", "y", retries=1)


def test_schema_constraints_are_enforced():
    class S(BaseModel):
        n: int

    with pytest.raises(StructuredError):
        generate_structured(Replies('{"n": "abc"}'), S, "x", "y", retries=0)


def test_rule_extractor_fields():
    t = rule_extract_ticket("My large fries were cold at store 7 and the staff were rude")
    assert t.category == "food_quality" and t.store_id == 7 and "large fries" in t.items


def test_allergen_always_escalates_even_if_model_says_low():
    low = ('{"category":"other","severity":"low","store_id":null,"items":[],"summary":"s",'
           '"needs_human":false}')
    out = extract_ticket("My daughter had an allergic reaction", Replies(low))
    assert out["ticket"]["severity"] == "high" and out["ticket"]["needs_human"] is True


def test_llm_failure_degrades_to_rules():
    class Boom(Replies):
        def chat(self, *a, **k):
            raise TimeoutError

    out = extract_ticket("wrong order, missing a burger", Boom())
    assert out["method"].startswith("rules") and out["ticket"]["category"] == "wrong_or_missing_item"


def test_endpoint_offline():
    r = TestClient(app).post("/extract/ticket", json={"text": "Waited ages at store 4 for cola"})
    assert r.status_code == 200
    body = r.json()
    assert body["ticket"]["category"] == "wait_time" and body["ticket"]["store_id"] == 4
    assert TestClient(app).post("/extract/ticket", json={"text": ""}).status_code == 422



def test_late_delivery_is_medium_severity_and_escalated():
    t = rule_extract_ticket(
        "Customer order: 2 classic burgers, 1 cola, no pickles. "
        "Customer reports the order arrived late."
    )
    assert t.category == "wait_time"
    assert t.severity == "medium"
    assert t.needs_human is True
    assert "classic burger" in t.items and "cola" in t.items
