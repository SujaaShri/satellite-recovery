import json

from spacecraft_state import SpacecraftState, apply_fault
from diagnosis_agent import diagnose
from recovery_planning_agent import plan_recovery
from recovery_procedure import RecoveryProcedure
from state_prediction import predict_trajectory
from frc_evaluator import evaluate_procedure


def run_experiment():

    print("=" * 60)
    print("SPACECRAFT RECOVERY EXPERIMENT")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Initial spacecraft state
    # ---------------------------------------------------------

    initial_state = SpacecraftState(
        battery_soc=80.0,
        thermal_margin=15.0,
        attitude_stability=95.0,
        comm_link_quality=90.0,
        payload_active=True,
        timestamp=0.0
    )

    print("\nINITIAL STATE")
    print(json.dumps(initial_state.as_dict(), indent=2))

    # ---------------------------------------------------------
    # 2. Inject fault
    # ---------------------------------------------------------

    state = apply_fault(initial_state, "battery_low")

    anomaly = (
        "Battery state of charge dropped from 80% to 60% "
        "while the payload remained active during an eclipse period."
    )

    print("\nOBSERVED ANOMALY")
    print(anomaly)

    print("\nFAULTED STATE")
    print(json.dumps(state.as_dict(), indent=2))

    # ---------------------------------------------------------
    # 3. RAG + Diagnosis
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("DIAGNOSIS")
    print("-" * 60)

    diagnosis = diagnose(anomaly, top_k_docs=3)

    print(json.dumps(diagnosis, indent=2))

    # ---------------------------------------------------------
    # 4. Recovery planning
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("RECOVERY PLANNING")
    print("-" * 60)

    plan = plan_recovery(
        diagnosis,
        state
    )

    print(json.dumps(plan, indent=2))

    if plan["selected_procedure"] is None:

        print("\nNO FEASIBLE RECOVERY PROCEDURE")

        return

    # ---------------------------------------------------------
    # 5. Simulate recovery
    # ---------------------------------------------------------

    procedure = RecoveryProcedure(
        **plan["selected_procedure"]
    )

    print("\nSELECTED PROCEDURE")
    print(procedure.procedure_id)

    print("\nACTIONS")
    for action in procedure.actions:
        print("-", action)

    trajectory = predict_trajectory(
        state,
        procedure,
        horizon_minutes=30,
        step_minutes=5
    )

    # ---------------------------------------------------------
    # 6. FRC evaluation
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("FRC EVALUATION")
    print("-" * 60)

    frc = evaluate_procedure(
        trajectory,
        [],
        threshold=2
    )

    print(json.dumps(frc, indent=2))

    # ---------------------------------------------------------
    # 7. Final decision
    # ---------------------------------------------------------

    print("\n" + "=" * 60)

    if frc["passes"]:
        print("FINAL DECISION: RECOVERY AUTHORIZED")
    else:
        print("FINAL DECISION: RECOVERY REJECTED")

    print("=" * 60)


if __name__ == "__main__":
    run_experiment()