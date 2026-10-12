from spacecraft_state import SpacecraftState
from frc_authority import evaluate_recovery_procedure


def main():

    state = SpacecraftState(
        battery_soc=22.0,
        thermal_margin=15.0,
        attitude_stability=60.0,
        comm_link_quality=90.0,
        payload_active=True,
        timestamp=0.0
    )

    procedure = {
        "procedure_id": "RP_03",
        "fault": "attitude_fault",
        "actions": [
            "reset_attitude_control_computer"
        ],
        "expected_effects": {
            "attitude_stability": 40.0
        },
        "preconditions": {
            "battery_soc": 20.0
        }
    }

    result = evaluate_recovery_procedure(
        spacecraft_state=state,
        procedure_dict=procedure,
        horizon_minutes=60,
        threshold=2
    )

    print("\n" + "=" * 70)
    print("FRC AUTHORITY TEST")
    print("=" * 70)

    print("\nPROCEDURE")
    print(procedure["procedure_id"])

    print("\nFRC RESULT")
    print(result)


if __name__ == "__main__":
    main()