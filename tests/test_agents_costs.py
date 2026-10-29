"""Tests for the agent loop, grounding validators and cost tracking, using a fake client.

These cover the guardrails that keep agent output tied to the data and the corpus:
citations must exist and have been retrieved, quotes must be verbatim, dollar amounts must
match candidate data, and invalid answers get exactly one chance to be fixed.
"""

import pytest
from helpers import FakeClient, final_response, path_series, tiny_chunks, tool_response

from rag_charts import config
from rag_charts.agents import (AgentError, AnalystOutput, ResearchOutput, Session,
                               VerifierOutput, analyst, build_annotations, check_prices,
                               researcher, strict_schema, verifier)
from rag_charts.costs import BudgetExceeded, CostTracker, call_cost
from rag_charts.patterns import detect, report_for_model
from rag_charts.retrieval import CorpusIndex
from types import SimpleNamespace


@pytest.fixture
def session():
    s = Session(index=CorpusIndex(tiny_chunks()))
    bars = path_series([(0, 90), (60, 120), (90, 105), (120, 120), (170, 95)])
    s.bars["3y"] = bars
    s.reports["3y"] = detect(bars, "daily")
    return s


def tracker(budget=5.0):
    return CostTracker("test", budget, log=False)


DEFINITION = {"pattern": "double top", "definition": "Two equal peaks.",
              "identification_rules": ["two peaks"], "confirmation": "support break",
              "implication": "bearish",
              "key_quote": "two roughly equal peaks and a moderate trough in between",
              "citations": ["sto-001-00"]}


# --- schema -----------------------------------------------------------------
def test_strict_schema_closes_every_object():
    schema = strict_schema(VerifierOutput)

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    found = list(objects(schema))
    assert found and all(o["additionalProperties"] is False for o in found)
    assert all(set(o["required"]) == set(o["properties"]) for o in found)
    assert "$ref" not in str(schema) and "$defs" not in schema


# --- researcher loop + grounding -------------------------------------------
def test_researcher_happy_path(session):
    client = FakeClient([
        tool_response(("search_corpus", {"query": "double top", "site": "any", "k": 3})),
        final_response({"definitions": [DEFINITION]}),
    ])
    t = tracker()
    out = researcher(["double top"], session, t, client)
    assert isinstance(out, ResearchOutput)
    assert len(client.requests) == 2
    req = client.requests[0]
    assert req["model"] == config.MODEL and req["fallbacks"] == "default"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert all(tool["strict"] for tool in req["tools"])
    assert t.summary()["by_tool"]["search_corpus"]["calls"] == 1


def test_citation_never_retrieved_is_rejected_then_fixed(session):
    client = FakeClient([
        final_response({"definitions": [DEFINITION]}),       # cites without searching
        tool_response(("search_corpus", {"query": "double top", "site": "any", "k": 3})),
        final_response({"definitions": [DEFINITION]}),
    ])
    researcher(["double top"], session, tracker(), client)
    feedback = client.requests[1]["messages"][-1]["content"]
    assert "never retrieved" in feedback


def test_fabricated_quote_fails_after_one_fix(session):
    bad = {**DEFINITION, "key_quote": "double tops always lead to crashes"}
    client = FakeClient([
        tool_response(("search_corpus", {"query": "double top", "site": "any", "k": 3})),
        final_response({"definitions": [bad]}),
        final_response({"definitions": [bad]}),
    ])
    with pytest.raises(AgentError, match="verbatim"):
        researcher(["double top"], session, tracker(), client)


def test_refusal_raises(session):
    client = FakeClient([final_response("{}", stop_reason="refusal")])
    with pytest.raises(AgentError, match="refused"):
        researcher(["double top"], session, tracker(), client)


def test_budget_stops_before_next_call(session):
    t = tracker(budget=0.0)
    with pytest.raises(BudgetExceeded):
        researcher(["double top"], session, t, FakeClient([]))


def test_tool_errors_are_returned_not_raised(session):
    client = FakeClient([
        tool_response(("get_chunk", {"chunk_id": "nope"})),
        tool_response(("search_corpus", {"query": "double top", "site": "any", "k": 3})),
        final_response({"definitions": [DEFINITION]}),
    ])
    # Researcher has no get_chunk tool, so this exercises the unknown-tool path.
    researcher(["double top"], session, tracker(), client)
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "unknown tool" in result["content"]


# --- analyst / verifier grounding -------------------------------------------
def _double_top(session):
    return next(c for c in session.reports["3y"]["candidates"] if c["type"] == "double_top")


def test_price_check_flags_invented_amounts(session):
    cand = _double_top(session)
    assert check_prices("Peaks at $120.00, trough $105", cand, [], "x") == []
    assert check_prices("Target $80", cand, [], "x")


def test_analyst_then_verifier_and_annotations(session):
    cand = _double_top(session)
    research = ResearchOutput.model_validate({"definitions": [DEFINITION]})
    session.retrieved.add("sto-001-00")
    selection = {"candidate_id": cand["id"], "label": "Double top (confirmed)",
                 "description": "Peaks near $120 with a trough at $105; confirmed by the break.",
                 "role": "bearish", "citations": ["sto-001-00"]}
    verdict = {"bias": "bearish", "confidence": "medium", "rationale": "Confirmed double top.",
               "citations": ["sto-001-00"]}
    view = report_for_model(session.reports["3y"])
    analysis = analyst("3y", "test", research, session, tracker(), FakeClient([
        final_response({"selections": [selection], "rejected": [], "verdict": verdict})]), view)
    assert isinstance(analysis, AnalystOutput)

    review = {**selection, "decision": "approve", "issues": ""}
    out = verifier("3y", "test", research, analysis, session, tracker(), FakeClient([
        final_response({"reviews": [review], "verdict": verdict,
                        "verdict_agrees_with_analyst": True, "notes": ""})]), view)
    annotations = build_annotations(out, "3y", session)
    assert annotations[0]["shapes"] == cand["shapes"]
    assert annotations[0]["citations"][0]["site"] == "stockcharts"


def test_verifier_must_review_every_selection(session):
    cand = _double_top(session)
    research = ResearchOutput.model_validate({"definitions": [DEFINITION]})
    session.retrieved.add("sto-001-00")
    sel = {"candidate_id": cand["id"], "label": "Double top", "description": "Two peaks.",
           "role": "bearish", "citations": ["sto-001-00"]}
    verdict = {"bias": "bearish", "confidence": "low", "rationale": "r", "citations": []}
    analysis = AnalystOutput.model_validate({"selections": [sel], "rejected": [], "verdict": verdict})
    empty = {"reviews": [], "verdict": verdict, "verdict_agrees_with_analyst": True, "notes": ""}
    with pytest.raises(AgentError, match="exactly once"):
        verifier("3y", "t", research, analysis, session, tracker(),
                 FakeClient([final_response(empty), final_response(empty)]),
                 report_for_model(session.reports["3y"]))


# --- costs -------------------------------------------------------------------
def test_call_cost_math():
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000,
                            cache_creation_input_tokens=0, cache_read_input_tokens=1_000_000)
    tokens, cost = call_cost("claude-opus-5-5", usage)
    assert tokens["output_tokens"] == 100_000
    assert cost == pytest.approx(4.00 + 2.00 + 0.20)


def test_cost_attribution_splits_parallel_tool_calls():
    t = tracker()
    resp = tool_response(("search_corpus", {}), ("get_candidate", {}))
    t.record_call("analyst", 1, resp, ["search_corpus", "get_candidate"])
    by_tool = t.summary()["by_tool"]
    assert by_tool["search_corpus"]["call_cost_usd"] == by_tool["get_candidate"]["call_cost_usd"]
