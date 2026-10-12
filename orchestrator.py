# orchestrator.py

import json
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END

from diagnosis_agent import diagnose
from recovery_agent import run_recovery_agent
from state_prediction import predict_trajectory
from frc_evaluator import evaluate_procedure
from recovery_procedure import RecoveryProcedure, PROCEDURE_LIBRARY
from spacecraft_state import SpacecraftState


class PipelineState(TypedDict):
    anomaly_description: str
    spacecraft_state: SpacecraftState

    diagnosis: Optional[dict]
    plan: Optional[dict]
    frc_result: Optional[dict]

    rejected_procedure_ids: List[str]
    replan_count: int
    max_replans: int

    final_decision: Optional[str]


# ---------------------------------------------------------
# DIAGNOSIS
# ---------------------------------------------------------

def node_diagnose(state: PipelineState) -> PipelineState:

    print("\n[1] DIAGNOSIS")

    diagnosis = diagnose(
        state["anomaly_description"],
        top_k_docs=3
    )

    state["diagnosis"] = diagnosis

    print(json.dumps(diagnosis, indent=2))

    return state


# ---------------------------------------------------------
# RECOVERY PLANNING
# ---------------------------------------------------------

def node_plan(state: PipelineState) -> PipelineState:

    print("\n[2] RECOVERY PLANNING AGENT")

    diagnosis = state["diagnosis"]

    result = run_recovery_agent(
        diagnosis=diagnosis,
        spacecraft_state=state["spacecraft_state"],
        excluded_procedure_ids=state["rejected_procedure_ids"]
    )

    selected_id = result.get(
        "selected_procedure_id"
    )

    selected_procedure = None

    if selected_id:

        selected_procedure = next(
            (
                p for p in PROCEDURE_LIBRARY
                if p.procedure_id == selected_id
            ),
            None
        )

    plan = {
        "selected_procedure": (
            selected_procedure.__dict__
            if selected_procedure
            else None
        ),
        "reasoning": result.get(
            "reasoning",
            ""
        ),
        "alternatives_considered": result.get(
            "alternatives_considered",
            []
        ),
        "rejected_procedures": result.get(
            "rejected_procedures",
            []
        ),
        "none_feasible": result.get(
            "none_feasible",
            False
        )
    }

    state["plan"] = plan

    print(
        json.dumps(
            plan,
            indent=2
        )
    )

    if selected_procedure:

        print(
            "\nSelected:",
            selected_procedure.procedure_id
        )

    else:

        print(
            "\nNo feasible procedure available."
        )

    return state


# ---------------------------------------------------------
# FRC
# ---------------------------------------------------------
# =========================================================
# 3. FRC ASSESSMENT
# =========================================================

def node_frc_assessment(state: PipelineState) -> PipelineState:

    print("\n[3] FRC EVALUATION")

    # -----------------------------------------------------
    # Get the current recovery plan
    # -----------------------------------------------------
    plan = state.get("plan")

    # -----------------------------------------------------
    # No procedure was selected
    # -----------------------------------------------------

    if not plan or not plan.get("selected_procedure"):

        frc_result = {
            "passes": False,
            "no_procedure_available": True,
            "reason": "No feasible recovery procedure available."
        }

        state["frc_result"] = frc_result

        print(
            json.dumps(
                frc_result,
                indent=2
            )
        )

        return state

    # -----------------------------------------------------
    # Convert selected procedure dictionary
    # into RecoveryProcedure object
    # -----------------------------------------------------

    procedure_dict = plan["selected_procedure"]

    procedure = RecoveryProcedure(
        **procedure_dict
    )

    # -----------------------------------------------------
    # Predict future spacecraft trajectory
    # -----------------------------------------------------

    # For the current Phase-B implementation,
    # evaluate the recovery over a 30-minute horizon.
    #
    # We can replace this with the actual communication
    # contact horizon once that component is finalized.

    horizon_minutes = 30

    trajectory = predict_trajectory(
        state["spacecraft_state"],
        procedure,
        horizon_minutes=horizon_minutes
    )

    # -----------------------------------------------------
    # Evaluate Future Recovery Capability
    # -----------------------------------------------------

    frc_result = evaluate_procedure(
        trajectory,
        PROCEDURE_LIBRARY,
        threshold=2
    )

    # -----------------------------------------------------
    # Validate FRC output
    # -----------------------------------------------------

    if frc_result is None:

        raise RuntimeError(
            "evaluate_procedure() returned None. "
            "Check frc_evaluator.py."
        )

    if not isinstance(frc_result, dict):

        raise TypeError(
            "evaluate_procedure() must return a dictionary, "
            f"but returned {type(frc_result).__name__}"
        )

    if "passes" not in frc_result:

        raise ValueError(
            "FRC result is missing the required 'passes' field."
        )

    # -----------------------------------------------------
    # Mark that a procedure was available
    # -----------------------------------------------------

    frc_result["no_procedure_available"] = False

    # -----------------------------------------------------
    # Store FRC result in LangGraph state
    # -----------------------------------------------------

    state["frc_result"] = frc_result

    # -----------------------------------------------------
    # Print result
    # -----------------------------------------------------

    print(
        json.dumps(
            frc_result,
            indent=2
        )
    )

    return state

# ---------------------------------------------------------
# ROUTING
# ---------------------------------------------------------

# ---------------------------------------------------------
# ROUTING
# ---------------------------------------------------------

def route_after_frc(state: PipelineState) -> str:

    frc = state.get("frc_result")

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if frc is None:
        state["final_decision"] = (
            "REJECTED: FRC evaluation did not produce a result"
        )
        return "give_up"

    # -----------------------------------------------------
    # CASE 1: FRC passed
    # -----------------------------------------------------

    if frc.get("passes", False):
        return "authorize"

    # -----------------------------------------------------
    # CASE 2: No feasible procedure was available
    # -----------------------------------------------------

    plan = state.get("plan")

    if (
        plan is None
        or plan.get("selected_procedure") is None
    ):

        state["final_decision"] = (
            "REJECTED: no feasible recovery procedure available"
        )

        return "give_up"

    # -----------------------------------------------------
    # CASE 3: Procedure existed but FRC rejected it
    # -----------------------------------------------------

    rejected_procedure = (
        plan["selected_procedure"]["procedure_id"]
    )

    if rejected_procedure not in state["rejected_procedure_ids"]:

        state["rejected_procedure_ids"].append(
            rejected_procedure
        )

    print(
        "\nFRC FAILED → REPLANNING"
    )

    print(
        "Rejected procedures:",
        state["rejected_procedure_ids"]
    )

    # -----------------------------------------------------
    # CASE 4: Maximum replans reached
    # -----------------------------------------------------

    if state["replan_count"] >= state["max_replans"]:

        state["final_decision"] = (
            "REJECTED: no recovery procedure passed "
            "FRC within the replan limit"
        )

        return "give_up"

    # -----------------------------------------------------
    # Perform another planning cycle
    # -----------------------------------------------------

    state["replan_count"] += 1

    return "replan"

# ---------------------------------------------------------
# AUTHORIZATION
# ---------------------------------------------------------

def node_authorize(state: PipelineState) -> PipelineState:

    procedure_id = (
        state["plan"]["selected_procedure"]["procedure_id"]
    )

    state["final_decision"] = (
        f"RECOVERY AUTHORIZED: {procedure_id}"
    )

    return state


# ---------------------------------------------------------
# GIVE UP
# ---------------------------------------------------------

def node_give_up(state: PipelineState) -> PipelineState:

    frc = state.get("frc_result", {})

    if frc.get("no_procedure_available"):

        state["final_decision"] = (
            "REJECTED: no feasible recovery procedure available"
        )

    else:

        rejected = state.get(
            "rejected_procedure_ids",
            []
        )

        state["final_decision"] = (
            "REJECTED: no recovery procedure passed "
            f"FRC after evaluating {len(rejected)} procedure(s)"
        )

    return state


# ---------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------

graph = StateGraph(PipelineState)

graph.add_node("diagnose", node_diagnose)
graph.add_node("plan", node_plan)
graph.add_node("frc_assessment", node_frc_assessment)
graph.add_node("authorize", node_authorize)
graph.add_node("give_up", node_give_up)

graph.set_entry_point("diagnose")

graph.add_edge(
    "diagnose",
    "plan"
)

graph.add_edge(
    "plan",
    "frc_assessment"
)

graph.add_conditional_edges(
    "frc_assessment",
    route_after_frc,
    {
        "authorize": "authorize",
        "replan": "plan",
        "give_up": "give_up"
    }
)

graph.add_edge(
    "authorize",
    END
)

graph.add_edge(
    "give_up",
    END
)

app = graph.compile()