from recovery_procedure import RecoveryProcedure, PROCEDURE_LIBRARY
from state_prediction import predict_trajectory
from frc_evaluator import evaluate_procedure


def evaluate_recovery_procedure(
    spacecraft_state,
    procedure_dict,
    horizon_minutes=60,
    threshold=2
):
    """
    Deterministic safety authority.

    The Recovery Agent proposes a procedure.
    This module independently simulates and evaluates it.

    The LLM has no authority to override the result.
    """

    if procedure_dict is None:
        return {
            "passes": False,
            "no_procedure_available": True,
            "reason": "No recovery procedure was proposed."
        }

    procedure = RecoveryProcedure(**procedure_dict)

    # --------------------------------------------------
    # Predict spacecraft trajectory
    # --------------------------------------------------

    trajectory = predict_trajectory(
        spacecraft_state,
        procedure,
        horizon_minutes=horizon_minutes
    )

    # --------------------------------------------------
    # Deterministic FRC evaluation
    # --------------------------------------------------

    frc_result = evaluate_procedure(
        trajectory,
        PROCEDURE_LIBRARY,
        threshold=threshold
    )

    if frc_result is None:
        raise RuntimeError(
            "FRC evaluator returned None."
        )

    if not isinstance(frc_result, dict):
        raise TypeError(
            "FRC evaluator must return a dictionary."
        )

    if "passes" not in frc_result:
        raise ValueError(
            "FRC result does not contain 'passes'."
        )

    # --------------------------------------------------
    # Add trajectory for auditability
    # --------------------------------------------------

    frc_result["trajectory"] = trajectory
    frc_result["procedure_id"] = procedure.procedure_id
    frc_result["no_procedure_available"] = False

    return frc_result