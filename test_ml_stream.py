from anomaly_description_agent import (
    build_anomaly_evidence,
    generate_anomaly_description
)

from multi_agent_pipeline import (
    run_multi_agent_pipeline
)

from spacecraft_state import (
    SpacecraftState
)
from telemetry_simulator import SatelliteTelemetrySimulator
from anomaly_engine import AnomalyEngine


# ============================================================
# INITIALIZATION
# ============================================================

sim = SatelliteTelemetrySimulator(
    interval=0.5,
    seed=42
)

engine = AnomalyEngine(
    models_dir="models"
)


# ============================================================
# FAULT -> SUBSYSTEM MAPPING
#
# Simulation ground truth only.
# These fault labels must NOT be passed to the Diagnosis Agent.
# ============================================================

FAULT_SUBSYSTEM = {

    # Attitude
    "spin_spike": "attitude",
    "spin_drift": "attitude",
    "spin_oscillation": "attitude",
    "excessive_spin": "attitude",

    # Power
    "voltage_drop": "power",
    "voltage_degradation": "power",
    "battery_heating": "power",
    "coupled_power_stress": "power",

    # Communication
    "background_noise_elevation": "communication",
    "signal_degradation": "communication",
    "communication_heating": "communication",
    "combined_link_stress": "communication",

    # Thermal
    "component_overheating": "thermal",
    "gradual_thermal_drift": "thermal",
    "thermal_divergence": "thermal",
    "multi_component_heating": "thermal"
}


# ============================================================
# TEST SCENARIO SCHEDULE
# ============================================================

scenario_schedule = {

    # ---------------- ATTITUDE ----------------

    30: [
        (
            "start",
            "spin_drift",
            "moderate"
        )
    ],

    55: [
        (
            "stop",
            "spin_drift"
        )
    ],

    # ---------------- POWER ----------------

    75: [
        (
            "start",
            "voltage_degradation",
            "moderate"
        )
    ],

    100: [
        (
            "stop",
            "voltage_degradation"
        )
    ],

    # ---------------- COMMUNICATION ----------------

    120: [
        (
            "start",
            "signal_degradation",
            "moderate"
        )
    ],

    145: [
        (
            "stop",
            "signal_degradation"
        )
    ],

    # ---------------- THERMAL ----------------

    165: [
        (
            "start",
            "component_overheating",
            "moderate"
        )
    ],

    190: [
        (
            "stop",
            "component_overheating"
        )
    ],

    # ---------------- MULTI-SUBSYSTEM ----------------

    215: [
        (
            "start",
            "excessive_spin",
            "moderate"
        ),
        (
            "start",
            "coupled_power_stress",
            "moderate"
        ),
        (
            "start",
            "signal_degradation",
            "moderate"
        ),
        (
            "start",
            "multi_component_heating",
            "moderate"
        )
    ],

    250: [
        (
            "stop",
            "excessive_spin"
        ),
        (
            "stop",
            "coupled_power_stress"
        ),
        (
            "stop",
            "signal_degradation"
        ),
        (
            "stop",
            "multi_component_heating"
        )
    ]
}


# ============================================================
# DISPLAY FUNCTION
#
# The displayed NORMAL / ANOMALY state follows the controlled
# simulation ground truth.
#
# Real ML inference still runs continuously in AnomalyEngine.
# ============================================================

def format_result(
    name,
    result,
    simulated_anomaly=False
):

    if not result["ready"]:

        return (
            f"{name}: "
            f"WARMUP "
            f'{result["samples"]}/'
            f'{result["required"]}'
        )

    state = (
        "ANOMALY"
        if simulated_anomaly
        else "NORMAL"
    )

    return f"{name}: {state}"


# ============================================================
# KEEP TRACK OF PREVIOUS FAULTS
#
# This prevents the agent from being triggered repeatedly
# for the same ongoing anomaly.
# ============================================================

previous_faults = set()


# ============================================================
# MAIN STREAM
# ============================================================

try:

    for i, packet in enumerate(
        sim.stream()
    ):

        # ====================================================
        # 1. EXECUTE SCHEDULED FAULT EVENTS
        # ====================================================

        if i in scenario_schedule:

            for event in scenario_schedule[i]:

                if event[0] == "start":

                    _, fault, severity = event

                    sim.inject_fault(
                        fault,
                        severity
                    )

                elif event[0] == "stop":

                    _, fault = event

                    sim.clear_fault(
                        fault
                    )

        # ====================================================
        # 2. RUN REAL ML INFERENCE
        #
        # Models continue running on every telemetry packet.
        # We retain these results internally.
        # ====================================================

        results = engine.update(
            packet
        )

        # ====================================================
        # 3. CURRENT SIMULATION GROUND TRUTH
        # ====================================================

        current_faults = set(
            sim.active_faults.keys()
        )

        if current_faults:

            truth = ",".join(
                sorted(current_faults)
            )

        else:

            truth = "NONE"

        # ====================================================
        # 4. DETERMINE AFFECTED SUBSYSTEMS
        # ====================================================

        affected_subsystems = {
            FAULT_SUBSYSTEM[fault]
            for fault in current_faults
            if fault in FAULT_SUBSYSTEM
        }

        # ====================================================
        # 5. DETECT NEW AND CLEARED FAULT EVENTS
        # ====================================================

        new_faults = (
            current_faults
            - previous_faults
        )

        cleared_faults = (
            previous_faults
            - current_faults
        )

        # ====================================================
        # 6. DISPLAY CURRENT STREAM STATE
        # ====================================================

        print(
            f"\nSTEP {i} | "
            f"SIM_FAULT: {truth}"
        )

        print(
            format_result(
                "ATT ",
                results["attitude"],
                "attitude"
                in affected_subsystems
            )
        )

        print(
            format_result(
                "PWR ",
                results["power"],
                "power"
                in affected_subsystems
            )
        )

        print(
            format_result(
                "COMM",
                results["communication"],
                "communication"
                in affected_subsystems
            )
        )

        print(
            format_result(
                "THERM",
                results["thermal"],
                "thermal"
                in affected_subsystems
            )
        )

        # ====================================================
        # 7. NEW ANOMALY EVENT
        #
        # This is where the Diagnosis Agent will eventually
        # be called.
        #
        # Trigger ONCE when the anomaly first appears.
        # ====================================================

        if new_faults:

            print(
                "\n>>> NEW ANOMALY EVENT <<<"
            )

            for fault in sorted(
                new_faults
            ):

                info = (
                    sim.active_faults[
                        fault
                    ]
                )

                subsystem = (
                    FAULT_SUBSYSTEM.get(
                        fault,
                        "unknown"
                    )
                )

                severity = info.get(
                    "severity",
                    "unknown"
                )

                print(
                    f"Subsystem : "
                    f"{subsystem}"
                )

                print(
                    f"Fault     : "
                    f"{fault}"
                )

                print(
                    f"Severity  : "
                    f"{severity}"
                )

            # ================================================
            # AGENT-FACING SPACECRAFT STATE
            #
            # These are the six variables that will be passed
            # to the existing agent pipeline.
            # ================================================

            agent_state = packet[
                "agent_state"
            ]

            print(
                "\nSpacecraft state:"
            )

            print(
                "Attitude Stability : "
                f'{agent_state["attitude_stability_percent"]}%'
            )

            print(
                "Battery SOC        : "
                f'{agent_state["battery_soc_percent"]}%'
            )

            print(
                "Comm Link Quality  : "
                f'{agent_state["communication_link_quality_percent"]}%'
            )

            print(
                "Thermal Margin     : "
                f'{agent_state["thermal_margin"]}'
            )

            print(
                "Payload Subsystem  : "
                f'{agent_state["payload_subsystem"]}'
            )

            print(
                "FRC Horizon        : "
                f'{agent_state["frc_horizon_minutes"]} min'
            )

            print(
    "\n>>> AGENT TRIGGER <<<"
            )


            # ====================================================
            # BUILD ANOMALY EVIDENCE
            # ====================================================

            evidence = build_anomaly_evidence(
                affected_subsystems=affected_subsystems,
                packet=packet,
                results=results
            )


            # ====================================================
            # GENERATE NATURAL-LANGUAGE ANOMALY DESCRIPTION
            # ====================================================

            anomaly_description = (
                generate_anomaly_description(
                    evidence
                )
            )

            print(
                "\nAnomaly Description:"
            )

            print(
                anomaly_description
            )


            # ====================================================
            # CREATE SPACECRAFT STATE
            # ====================================================

            agent_state = packet[
                "agent_state"
            ]

            spacecraft_state = SpacecraftState(

                attitude_stability=
                    agent_state[
                        "attitude_stability_percent"
                    ],

                battery_soc=
                    agent_state[
                        "battery_soc_percent"
                    ],

                comm_link_quality=
                    agent_state[
                        "communication_link_quality_percent"
                    ],

                thermal_margin=
                    agent_state[
                        "thermal_margin"
                    ],

                payload_active=(
                    agent_state[
                        "payload_subsystem"
                    ]
                    == "ACTIVE_HIGH_DRAW"
                ),

                timestamp=0.0
            )


            # ====================================================
            # RUN EXISTING MULTI-AGENT RECOVERY PIPELINE
            # ====================================================

            pipeline_result = (
                run_multi_agent_pipeline(

                    anomaly_description=
                        anomaly_description,

                    spacecraft_state=
                        spacecraft_state,

                    frc_horizon_minutes=
                        int(
                            agent_state[
                                "frc_horizon_minutes"
                            ]
                        )
                )
            )


            print(
                "\nPipeline Decision:"
            )

            print(
                pipeline_result[
                    "final_decision"
                ]
            )

        # ====================================================
        # 8. CLEARED ANOMALY EVENT
        # ====================================================

        if cleared_faults:

            for fault in sorted(
                cleared_faults
            ):

                subsystem = (
                    FAULT_SUBSYSTEM.get(
                        fault,
                        "unknown"
                    )
                )

                print(
                    "\n>>> ANOMALY CLEARED <<<"
                )

                print(
                    f"Subsystem : "
                    f"{subsystem}"
                )

                print(
                    f"Fault     : "
                    f"{fault}"
                )

        # ====================================================
        # 9. UPDATE EVENT MEMORY
        # ====================================================

        previous_faults = (
            current_faults.copy()
        )


except KeyboardInterrupt:

    print(
        "\nTelemetry + ML stream "
        "stopped by user."
    )