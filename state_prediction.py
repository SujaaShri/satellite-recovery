


from recovery_procedure import RecoveryProcedure

from spacecraft_state import SpacecraftState


def predict_trajectory(
    state,
    procedure,
    horizon_minutes,
    step_minutes=5
):
    """
    Predict spacecraft state over the recovery horizon.

    This is a simplified deterministic spacecraft dynamics model.
    The purpose is to evaluate whether a proposed recovery remains
    inside the Future Recovery Capability (FRC) safety corridor.
    """

    trajectory = []

    current = SpacecraftState(**state.as_dict())

    # Apply the immediate recovery action first.
    current = apply_procedure_effects(current, procedure)

    # Store initial post-action state.
    trajectory.append({
        "time": 0,
        **current.as_dict()
    })

    # ---------------------------------------------------------
    # FUTURE DYNAMICS
    # ---------------------------------------------------------

    for t in range(step_minutes, int(horizon_minutes) + 1, step_minutes):

        next_state = SpacecraftState(**current.as_dict())

        # -----------------------------------------------------
        # Normal spacecraft drift
        # -----------------------------------------------------

        # Battery naturally decreases with time.
        next_state.battery_soc -= 0.25

        # Small thermal drift.
        next_state.thermal_margin -= 0.05

        # Small attitude drift.
        next_state.attitude_stability -= 0.10

        # Small communication drift.
        next_state.comm_link_quality -= 0.10

        # -----------------------------------------------------
        # Recovery-specific dynamics
        # -----------------------------------------------------

        if procedure.procedure_id == "RP_03":
            """
            Controller reset consumes power while the spacecraft
            attempts to reacquire attitude control.

            When battery is marginal, this causes the predicted
            future battery state to cross the FRC safety boundary.
            """

            next_state.battery_soc -= 1.5

            # Attitude gradually improves after reset.
            next_state.attitude_stability += 1.5

        elif procedure.procedure_id == "RP_07":
            """
            Safe mode is deliberately conservative.

            It reduces subsystem activity and therefore preserves
            battery while providing modest attitude stabilization.
            """

            next_state.battery_soc += 0.5
            next_state.attitude_stability += 0.5

        elif procedure.procedure_id == "RP_01":

            # Payload has been disabled, so battery decline is reduced.
            next_state.battery_soc += 0.15

        elif procedure.procedure_id == "RP_02":

            # Safe mode conserves power.
            next_state.battery_soc += 0.50

            # But attitude performance is temporarily degraded.
            next_state.attitude_stability -= 0.10

        elif procedure.procedure_id == "RP_05":

            # Heater consumes some battery.
            next_state.battery_soc -= 0.50

            # Thermal recovery.
            next_state.thermal_margin += 0.80

        elif procedure.procedure_id == "RP_04":

            # Backup communication stabilizes link quality.
            next_state.comm_link_quality += 1.0

        elif procedure.procedure_id == "RP_06":

            # Lower transmit power saves battery.
            next_state.battery_soc += 0.25

        # -----------------------------------------------------
        # Clamp physical values
        # -----------------------------------------------------

        next_state.battery_soc = max(
            0.0,
            min(100.0, next_state.battery_soc)
        )

        next_state.thermal_margin = max(
            0.0,
            min(100.0, next_state.thermal_margin)
        )

        next_state.attitude_stability = max(
            0.0,
            min(100.0, next_state.attitude_stability)
        )

        next_state.comm_link_quality = max(
            0.0,
            min(100.0, next_state.comm_link_quality)
        )

        next_state.timestamp = float(t)

        trajectory.append({
            "time": t,
            **next_state.as_dict()
        })

        current = next_state

    return trajectory


def apply_procedure_effects(state, procedure):

    new_state = SpacecraftState(**state.as_dict())

    for variable, effect in procedure.expected_effects.items():

        if hasattr(new_state, variable):

            current_value = getattr(new_state, variable)

            setattr(
                new_state,
                variable,
                current_value + effect
            )

    # Specific action effects
    if "disable_nonessential_payload" in procedure.actions:
        new_state.payload_active = False

    if "enter_safe_mode" in procedure.actions:
        new_state.payload_active = False

    return new_state