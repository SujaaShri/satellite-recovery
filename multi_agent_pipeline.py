# multi_agent_pipeline.py

import json

from diagnosis_agent_react import run_diagnosis_agent
from recovery_agent import run_recovery_agent

from spacecraft_state import SpacecraftState
from recovery_procedure import RecoveryProcedure, PROCEDURE_LIBRARY
from state_prediction import predict_trajectory
from frc_evaluator import evaluate_procedure


# ============================================================
# FRC AUTHORITY
# ============================================================

def run_frc_authority(
    spacecraft_state,
    procedure_id,
    horizon_minutes=60
):
    """
    Independent FRC safety authority.

    The Recovery ReAct Agent proposes a procedure.
    This function independently simulates and evaluates it.
    """

    # --------------------------------------------------------
    # Find selected procedure
    # --------------------------------------------------------

    procedure = None

    for proc in PROCEDURE_LIBRARY:

        if proc.procedure_id == procedure_id:
            procedure = proc
            break

    # --------------------------------------------------------
    # Unknown procedure
    # --------------------------------------------------------

    if procedure is None:

        return {
            "passes": False,
            "no_procedure_available": True,
            "procedure_id": procedure_id,
            "reason": (
                f"Unknown recovery procedure: "
                f"{procedure_id}"
            )
        }

    # --------------------------------------------------------
    # Predict future spacecraft trajectory
    # --------------------------------------------------------

    trajectory = predict_trajectory(
        spacecraft_state,
        procedure,
        horizon_minutes=horizon_minutes
    )

    # --------------------------------------------------------
    # Independent FRC evaluation
    # --------------------------------------------------------

    frc_result = evaluate_procedure(
        trajectory,
        PROCEDURE_LIBRARY,
        threshold=2
    )

    # --------------------------------------------------------
    # Validate evaluator output
    # --------------------------------------------------------

    if frc_result is None:

        raise RuntimeError(
            "evaluate_procedure() returned None."
        )

    if not isinstance(frc_result, dict):

        raise TypeError(
            "evaluate_procedure() must return a dictionary."
        )

    if "passes" not in frc_result:

        raise ValueError(
            "FRC result is missing the 'passes' field."
        )

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    frc_result["procedure_id"] = procedure_id
    frc_result["no_procedure_available"] = False
    frc_result["trajectory"] = trajectory

    return frc_result


# ============================================================
# MULTI-AGENT PIPELINE
# ============================================================

def run_multi_agent_pipeline(
    anomaly_description,
    spacecraft_state,
    frc_horizon_minutes=60
):

    print("\n")
    print("=" * 80)
    print("SPACECRAFT MULTI-AGENT RECOVERY SYSTEM")
    print("=" * 80)

    # ========================================================
    # 1. DIAGNOSIS REACT AGENT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("1. DIAGNOSIS ReAct AGENT")
    print("=" * 70)

    diagnosis = run_diagnosis_agent(
        anomaly_description=anomaly_description,
        spacecraft_state=spacecraft_state
    )

    print("\nDIAGNOSIS RESULT")
    print(
        json.dumps(
            diagnosis,
            indent=2
        )
    )

    # ========================================================
    # 2. RECOVERY ReAct AGENT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("2. RECOVERY ReAct AGENT")
    print("=" * 70)

    recovery = run_recovery_agent(
        diagnosis=diagnosis,
        spacecraft_state=spacecraft_state
    )

    print("\nRECOVERY RESULT")
    print(
        json.dumps(
            recovery,
            indent=2
        )
    )

    # ========================================================
    # Check whether Recovery Agent found anything
    # ========================================================

    selected_procedure_id = recovery.get(
        "selected_procedure_id"
    )

    if not selected_procedure_id:

        return {
            "diagnosis": diagnosis,
            "recovery": recovery,
            "frc": {
                "passes": False,
                "no_procedure_available": True
            },
            "final_decision": (
                "REJECTED: no feasible recovery procedure available"
            )
        }

    # ========================================================
    # 3. FRC SAFETY AUTHORITY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("3. FRC SAFETY AUTHORITY")
    print("=" * 70)

    print(
        f"\nEvaluating procedure: "
        f"{selected_procedure_id}"
    )

    frc_result = run_frc_authority(
        spacecraft_state=spacecraft_state,
        procedure_id=selected_procedure_id,
        horizon_minutes=frc_horizon_minutes
    )

    print("\nFRC RESULT")

    print(
        json.dumps(
            frc_result,
            indent=2
        )
    )

    # ========================================================
    # 4. FINAL DECISION
    # ========================================================

    if frc_result["passes"]:

        final_decision = (
            f"RECOVERY AUTHORIZED: "
            f"{selected_procedure_id}"
        )

    else:

        final_decision = (
            f"RECOVERY REJECTED BY FRC: "
            f"{selected_procedure_id}"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    result = {

        "diagnosis": diagnosis,

        "recovery": recovery,

        "frc": frc_result,

        "final_decision": final_decision
    }

    print("\n")
    print("=" * 80)
    print("FINAL MULTI-AGENT DECISION")
    print("=" * 80)

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    return result