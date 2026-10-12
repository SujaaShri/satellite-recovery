import json

from recovery_agent import run_recovery_agent
from spacecraft_state import SpacecraftState


def main():

    state = SpacecraftState(
        battery_soc=22.0,
        thermal_margin=15.0,
        attitude_stability=60.0,
        comm_link_quality=90.0,
        payload_active=True,
        timestamp=0.0
    )

    diagnosis = {
        "fault_type": "attitude_fault",
        "evidence": (
            "Attitude stability dropped significantly "
            "while battery remains marginal."
        ),
        "confidence": 0.94,
        "candidate_procedure_ids": [
            "RP_03"
        ]
    }

    print("=" * 70)
    print("RECOVERY REACT AGENT TEST")
    print("=" * 70)

    result = run_recovery_agent(
        diagnosis=diagnosis,
        spacecraft_state=state,
        excluded_procedure_ids=[]
    )

    print("\n" + "=" * 70)
    print("FINAL AGENT RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":
    main()