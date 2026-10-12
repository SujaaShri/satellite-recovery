import json

from multi_agent_pipeline import run_multi_agent_pipeline
from spacecraft_state import SpacecraftState


def get_float(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()

    if value == "":
        return float(default)

    try:
        return float(value)
    except ValueError:
        print("Invalid number. Using default.")
        return float(default)


def get_bool(prompt, default=True):
    value = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()

    if value == "":
        return default

    return value in ["y", "yes", "true", "1"]


def main():

    print("=" * 80)
    print("SATELLITE MULTI-AGENT RECOVERY SYSTEM")
    print("=" * 80)

    print("\nEnter spacecraft telemetry / observed anomaly.")
    print("Press ENTER to use the default value shown.\n")

    # ---------------------------------------------------------
    # ANOMALY DESCRIPTION
    # ---------------------------------------------------------

    anomaly = input(
        "Observed anomaly description: "
    ).strip()

    if not anomaly:
        anomaly = (
            "Attitude stability dropped from 95% to 60% "
            "while battery state of charge is marginal at 22%."
        )

    # ---------------------------------------------------------
    # SPACECRAFT STATE
    # ---------------------------------------------------------

    print("\n--- SPACECRAFT TELEMETRY ---")

    battery_soc = get_float(
        "Battery SOC (%)",
        22
    )

    thermal_margin = get_float(
        "Thermal margin (%)",
        15
    )

    attitude_stability = get_float(
        "Attitude stability (%)",
        60
    )

    comm_link_quality = get_float(
        "Communication link quality (%)",
        90
    )

    payload_active = get_bool(
        "Payload active?",
        True
    )

    # ---------------------------------------------------------
    # CREATE SPACECRAFT STATE
    # ---------------------------------------------------------

    state = SpacecraftState(
        battery_soc=battery_soc,
        thermal_margin=thermal_margin,
        attitude_stability=attitude_stability,
        comm_link_quality=comm_link_quality,
        payload_active=payload_active,
        timestamp=0.0
    )

    print("\n" + "=" * 80)
    print("INPUT TELEMETRY")
    print("=" * 80)

    print(
        json.dumps(
            state.as_dict(),
            indent=2
        )
    )

    print("\n" + "=" * 80)
    print("RUNNING MULTI-AGENT RECOVERY PIPELINE")
    print("=" * 80)

    # ---------------------------------------------------------
    # RUN COMPLETE PIPELINE
    # ---------------------------------------------------------

    result = run_multi_agent_pipeline(
        anomaly_description=anomaly,
        spacecraft_state=state,
        frc_horizon_minutes=60
    )

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL MULTI-AGENT DECISION")
    print("=" * 80)

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":
    main()