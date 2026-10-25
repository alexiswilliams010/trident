import json

import pytest

from core.connections.inference import prepare
from core.connections.presentation import comparison
from core.connections.assessment import interpret
from core.connections.questions import recipient_questions, relationship_questions
from tests.test_connection_assessment import build_questions, minimal_packet, response_for


def test_personalized_question_and_repository_labels():
    raw = b"fetch()"
    site = dict(
        id="site", path="code.py", kind="call", start_byte=0, end_byte=7, metadata={"name": "fetch"}
    )
    state, questions, _ = prepare(site, raw, site, raw)
    data = json.loads(state)
    data["source"]["repository"] = "client-project"
    data["target"]["repository"] = "service-project"
    text, questions = comparison(json.dumps(data), questions)
    assert "client-project · code.py:1" in text
    assert "service-project · code.py:1" in text
    assert "neptune" not in text and "trident" not in text
    assert "call fetch at code.py:1" in questions["condition"]["instructions"]
    assert text.index("ENDPOINT COMPARISON") < text.index("\nSOURCE\n")


@pytest.mark.parametrize("topology", ["forward", "reverse", "shared_resource"])
def test_intermediate_recipient_blocks_edges_despite_supported_topology(topology):
    packet = minimal_packet()
    questions = recipient_questions(relationship_questions(build_questions(packet)))
    answers = response_for(
        questions, {"condition": "intermediate", "topology": topology, "resource_anchor": "a"}
    )["answers"]
    proposal = interpret(packet, {"source_site_id": "source", "target_site_id": "target"}, answers)
    assert proposal["condition"] == "intermediate"
    assert proposal["outcome"] == "insufficient_evidence"
    assert "intermediate_recipient" in proposal["abstention_reasons"]
    assert not proposal["edges"]


@pytest.mark.parametrize(
    "topology,recipient,supported",
    [
        ("forward", "exact_target", True),
        ("forward", "shared_resource", False),
        ("shared_resource", "shared_resource", True),
        ("forward", "different_destination", False),
        ("forward", "unresolved", False),
        ("indirect", "exact_target", False),
    ],
)
def test_recipient_and_topology_consistency(topology, recipient, supported):
    packet = minimal_packet()
    questions = recipient_questions(relationship_questions(build_questions(packet)))
    answers = response_for(
        questions, {"condition": recipient, "topology": topology, "resource_anchor": "a"}
    )["answers"]
    proposal = interpret(packet, {"source_site_id": "source", "target_site_id": "target"}, answers)
    assert (proposal["outcome"] == "supported") is supported
    assert bool(proposal["edges"]) is supported
