"""
test_pipeline.py
Run with: pytest test_pipeline.py -v

Tests split into two groups:
  - OFFLINE tests (mocked Groq calls): run every time, no API cost, no network.
  - LIVE tests (marked): only run with `pytest test_pipeline.py -v -m live`,
    actually hit Groq, cost real tokens, require GROQ_API_KEY set.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from fault_knowledge_base import FAULT_KNOWLEDGE_BASE
from rag_retriever import FaultRetriever
from spacecraft_state import SpacecraftState, apply_fault
from recovery_procedure import PROCEDURE_LIBRARY, check_preconditions


# ---------- KB integrity ----------

def test_kb_no_missing_fault_type():
    for doc in FAULT_KNOWLEDGE_BASE:
        assert "fault_type" in doc and doc["fault_type"], f"{doc['doc_id']} missing fault_type"

def test_kb_fault_types_valid():
    valid = {"battery_low", "thermal_excursion", "attitude_fault", "comm_degradation"}
    for doc in FAULT_KNOWLEDGE_BASE:
        assert doc["fault_type"] in valid

def test_kb_related_procedures_exist():
    valid_ids = {p.procedure_id for p in PROCEDURE_LIBRARY}
    for doc in FAULT_KNOWLEDGE_BASE:
        for pid in doc["related_procedures"]:
            assert pid in valid_ids, f"{doc['doc_id']} references unknown procedure {pid}"


# ---------- Retrieval (real embeddings, no API key needed) ----------

@pytest.fixture(scope="module")
def retriever():
    return FaultRetriever()

def test_retrieval_returns_correct_fault_domain(retriever):
    results = retriever.retrieve("battery state of charge dropping while payload runs", top_k=3)
    assert len(results) == 3
    # top hit should be battery-related for an unambiguous battery query
    assert results[0]["fault_type"] == "battery_low"

def test_retrieval_filter_by_fault_type(retriever):
    results = retriever.retrieve("something going wrong", top_k=5, fault_type_filter="thermal_excursion")
    assert all(r["fault_type"] == "thermal_excursion" for r in results)


# ---------- Feasibility logic (pure Python, no LLM) ----------

def test_precondition_check_blocks_when_battery_low():
    state = SpacecraftState(battery_soc=5.0, thermal_margin=15.0, attitude_stability=95.0,
                             comm_link_quality=90.0, payload_active=True, timestamp=0.0)
    rp03 = next(p for p in PROCEDURE_LIBRARY if p.procedure_id == "RP_03")
    assert check_preconditions(state, rp03) is False  # requires battery_soc >= 20

def test_precondition_check_passes_when_sufficient():
    state = SpacecraftState(battery_soc=50.0, thermal_margin=15.0, attitude_stability=95.0,
                             comm_link_quality=90.0, payload_active=True, timestamp=0.0)
    rp03 = next(p for p in PROCEDURE_LIBRARY if p.procedure_id == "RP_03")
    assert check_preconditions(state, rp03) is True

def test_apply_fault_battery_low():
    state = SpacecraftState(battery_soc=80.0, thermal_margin=15.0, attitude_stability=95.0,
                             comm_link_quality=90.0, payload_active=True, timestamp=0.0)
    faulted = apply_fault(state, "battery_low")
    assert faulted.battery_soc == 60.0


# ---------- Diagnosis agent (mocked Groq — tests JSON parsing/wiring, not model quality) ----------

def _mock_groq_response(payload: dict):
    mock_message = MagicMock()
    mock_message.content = json.dumps(payload)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response

def test_diagnose_parses_valid_json():
    from diagnosis_agent import diagnose
    fake_payload = {
        "fault_type": "battery_low", "evidence": "mocked", "confidence": 0.9,
        "supporting_doc_ids": ["KB_01"], "candidate_procedure_ids": ["RP_01"],
    }
    with patch("diagnosis_agent.client.chat.completions.create", return_value=_mock_groq_response(fake_payload)):
        result = diagnose("battery dropping steadily")
        assert result["fault_type"] == "battery_low"
        assert "retrieved_doc_ids" in result  # confirms retrieval actually ran, not just mocked LLM

def test_diagnose_raises_on_invalid_json():
    from diagnosis_agent import diagnose
    bad_response = MagicMock()
    bad_response.choices = [MagicMock(message=MagicMock(content="not json"))]
    with patch("diagnosis_agent.client.chat.completions.create", return_value=bad_response):
        with pytest.raises(json.JSONDecodeError):
            diagnose("battery dropping")


# ---------- Planning agent: the rejection-exclusion bug fix, tested directly ----------

def test_plan_recovery_excludes_rejected_procedure():
    from recovery_planning_agent import plan_recovery
    state = SpacecraftState(battery_soc=80.0, thermal_margin=15.0, attitude_stability=95.0,
                             comm_link_quality=90.0, payload_active=True, timestamp=0.0)
    diagnosis = {"fault_type": "battery_low", "evidence": "mocked", "confidence": 0.9}

    fake_decision = {"selected_procedure_id": "RP_02", "reasoning": "mocked",
                      "alternatives_considered": [], "none_feasible": False}
    with patch("recovery_planning_agent.client.chat.completions.create",
               return_value=_mock_groq_response(fake_decision)):
        result = plan_recovery(diagnosis, state, excluded_procedure_ids=["RP_01"])
        # RP_01 must not even appear in the feasibility report the LLM was shown
        reported_ids = [r["procedure_id"] for r in result["feasibility_report"]]
        assert "RP_01" not in reported_ids
        assert result["selected_procedure"]["procedure_id"] == "RP_02"

def test_plan_recovery_rejects_llm_picking_infeasible_procedure():
    from recovery_planning_agent import plan_recovery
    state = SpacecraftState(battery_soc=5.0, thermal_margin=15.0, attitude_stability=10.0,
                             comm_link_quality=90.0, payload_active=True, timestamp=0.0)
    diagnosis = {"fault_type": "attitude_fault", "evidence": "mocked", "confidence": 0.9}
    # RP_03 requires battery_soc >= 20, state has 5 -- infeasible.
    # If the LLM (mocked here) tries to select it anyway, code must raise, not silently comply.
    fake_decision = {"selected_procedure_id": "RP_03", "reasoning": "mocked",
                      "alternatives_considered": [], "none_feasible": False}
    with patch("recovery_planning_agent.client.chat.completions.create",
               return_value=_mock_groq_response(fake_decision)):
        with pytest.raises(ValueError, match="infeasible"):
            plan_recovery(diagnosis, state)


# ---------- LIVE integration test (real API calls — run explicitly, costs tokens) ----------

@pytest.mark.live
def test_live_diagnose_and_plan_end_to_end():
    from diagnosis_agent import diagnose
    from recovery_planning_agent import plan_recovery
    state = SpacecraftState(battery_soc=80.0, thermal_margin=15.0, attitude_stability=95.0,
                             comm_link_quality=90.0, payload_active=True, timestamp=10.0)
    state = apply_fault(state, "battery_low")
    diagnosis = diagnose("battery_soc dropped from 80 to 60 over 40 minutes while payload_active remained true")
    assert diagnosis["fault_type"] in {"battery_low", "thermal_excursion", "attitude_fault", "comm_degradation"}
    plan = plan_recovery(diagnosis, state)
    print("\nLIVE diagnosis:", diagnosis)
    print("LIVE plan:", plan)
    assert plan["selected_procedure"] is not None or plan["reasoning"]