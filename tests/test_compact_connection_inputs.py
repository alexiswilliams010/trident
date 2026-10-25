import json

import pytest

from core.connections.inference import endpoint, prepare, POLICY


def site(content, expression, **updates):
    start = content.index(expression)
    return dict(
        id="internal-id",
        start_byte=start,
        end_byte=start + len(expression),
        path="module.py",
        kind="call",
        metadata={"name": "send"},
        **updates,
    )


def test_exact_endpoint_excludes_neighboring_calls_and_provenance():
    content = b"from service import send\nother()\nsend(payload)\nneighbor()"
    source = site(content, b"send(payload)")
    state, questions, packet = prepare(source, content, source, content)
    parsed = json.loads(state)
    assert parsed["source"]["exact_code"] == "send(payload)"
    assert "other()" in parsed["source"]["context_before"]
    assert "neighbor()" in parsed["source"]["context_after"]
    assert "internal-id" not in state
    assert parsed["instructions"] == POLICY
    assert all(POLICY not in q["instructions"] for q in questions.values())
    assert packet["excerpts"][0]["site_id"] == "internal-id"
    assert len(questions) == 9


def test_utf8_truncation_and_explicit_clipping():
    content = 'before\nf("éééé")\nafter'.encode()
    result = endpoint(site(content, 'f("éééé")'.encode()), content, code_bytes=4)
    assert result["exact_code"] == 'f("'
    assert result["exact_code_clipped"]
    assert result["line"] == 2
    with pytest.raises(ValueError, match="outside"):
        endpoint({**site(content, b"f("), "end_byte": len(content) + 1}, content)


def test_plaintext_preserves_code_and_orders_sections():
    from core.connections.inference import plaintext

    content = b'from service import send\nsend("```")\nneighbor()'
    source = site(content, b'send("```")')
    state, _, _ = prepare(source, content, source, content)
    text = plaintext(state)
    for key in ("exact_code", "context_before", "context_after"):
        assert json.loads(state)["source"][key] in text
    assert '````\nsend("```")\n````' in text
    assert text.index("SOURCE") < text.index("TARGET")
    assert (
        text.index("Exact code:")
        < text.index("Preceding context:")
        < text.index("Following context:")
    )
    assert "internal-id" not in text
    assert "Only Exact code identifies the endpoint" in text


def test_additional_context_and_resource_citations():
    from core.connections.inference import plaintext, remap_proposal

    content = b"send(payload)"
    source = site(content, content)
    extra = {
        "evidence_id": "stored-config-id",
        "path": "routing.conf",
        "start_byte": 0,
        "end_byte": 12,
        "excerpt": "topic=orders",
    }
    state, questions, packet = prepare(source, content, source, content, additional=[extra])
    assert "topic=orders" in plaintext(state)
    assert "stored-config-id" not in plaintext(state)
    assert "context_1" in questions["resource_anchor"]["criteria"]
    assert packet["excerpts"][-1]["evidence_id"] == "context_1"
    proposal = remap_proposal(
        {
            "evidence_roles": {"source": "support", "target": "support", "context_1": "context"},
            "edges": [{"resource_evidence_id": "source", "verb": "PRODUCES"}],
        },
        {"source": "citation-a", "target": "citation-b", "context_1": "citation-a"},
    )
    assert proposal["evidence_roles"] == {"citation-a": "support", "citation-b": "support"}
    assert proposal["edges"][0]["resource_evidence_id"] == "citation-a"
