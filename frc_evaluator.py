# frc_evaluator.py

def evaluate_procedure(
    trajectory,
    procedure_library,
    threshold=2
):
    """
    Evaluate whether a recovery procedure keeps the spacecraft
    inside a safe/recoverable operating region.

    threshold = number of violated safety dimensions allowed.
    """

    violations = []

    for state in trajectory:

        state_violations = []

        if state["battery_soc"] < 20:
            state_violations.append("battery_soc")

        if state["thermal_margin"] < 10:
            state_violations.append("thermal_margin")

        if state["attitude_stability"] < 50:
            state_violations.append("attitude_stability")

        if state["comm_link_quality"] < 30:
            state_violations.append("comm_link_quality")

        if state_violations:
            violations.append({
                "time": state["time"],
                "violations": state_violations
            })

    # Count distinct safety dimensions violated
    unique_violations = set()

    for v in violations:
        unique_violations.update(v["violations"])

    passes = len(unique_violations) < threshold

    return {
        "passes": passes,
        "violated_dimensions": list(unique_violations),
        "violation_count": len(unique_violations),
        "max_violation_count": threshold,
        "final_state": trajectory[-1],
        "trajectory_length": len(trajectory),
    }