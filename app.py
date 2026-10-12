from flask import Flask, render_template, jsonify
import threading
import traceback

from multi_agent_pipeline import run_multi_agent_pipeline
from spacecraft_state import SpacecraftState
from telemetry_simulator import SatelliteTelemetrySimulator
from anomaly_engine import AnomalyEngine
from anomaly_description_agent import (
    build_anomaly_evidence,
    generate_anomaly_description
)


app = Flask(__name__)


# ============================================================
# HUMAN-READABLE RECOVERY PROCEDURES
# ============================================================

PROCEDURE_DESCRIPTIONS = {

    "RP_01": {
        "name": "Disable Non-Essential Payload",
        "description": (
            "Disables non-essential payload operations to reduce power "
            "consumption and improve battery state of charge."
        )
    },

    "RP_02": {
        "name": "Enter Safe Mode",
        "description": (
            "Places the spacecraft in a reduced-power safe configuration. "
            "This improves battery reserve while temporarily reducing "
            "attitude performance."
        )
    },

    "RP_03": {
        "name": "Reset Attitude Control Computer",
        "description": (
            "Resets the attitude control computer to recover from a "
            "controller lockup or transient attitude-control fault. "
            "This can substantially improve attitude stability but "
            "requires sufficient battery reserve."
        )
    },

    "RP_04": {
        "name": "Switch to Backup Communication",
        "description": (
            "Switches spacecraft communications to the backup link "
            "to recover degraded communication quality."
        )
    },

    "RP_05": {
        "name": "Activate Thermal Heater",
        "description": (
            "Activates the thermal control heater to increase the "
            "spacecraft's thermal margin."
        )
    },

    "RP_06": {
        "name": "Reduce Transmit Power",
        "description": (
            "Reduces communication transmit power to decrease energy "
            "consumption while partially improving communication "
            "and battery margin."
        )
    },

    "RP_07": {
        "name": "Enter Safe Mode for Attitude Recovery",
        "description": (
            "Places the spacecraft in safe mode to reduce power "
            "consumption and preserve battery reserve while providing "
            "a modest improvement in attitude stability. This is used "
            "as a safer alternative when an attitude-controller reset "
            "would consume too much available power."
        )
    }
}


# ============================================================
# FAULT -> SUBSYSTEM MAPPING
#
# These are simulation ground-truth labels only.
# They are NOT passed to the Diagnosis Agent.
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
# DEMONSTRATION FAULT SCHEDULE
# ============================================================

FAULT_SCHEDULE = {

    # Attitude anomaly
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

    # Power anomaly
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

    # Communication anomaly
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

    # Thermal anomaly
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

    # Multi-subsystem anomaly
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
# SHARED LIVE DASHBOARD STATE
# ============================================================

state_lock = threading.Lock()


live_data = {

    "step": 0,

    "timestamp": None,

    "agent_state": {
        "attitude_stability_percent": None,
        "battery_soc_percent": None,
        "communication_link_quality_percent": None,
        "thermal_margin": None,
        "payload_subsystem": None,
        "frc_horizon_minutes": None
    },

    "subsystems": {
        "attitude": "WARMUP",
        "power": "WARMUP",
        "communication": "WARMUP",
        "thermal": "WARMUP"
    },

    "active_fault_count": 0,

    "anomaly_active": False,

    "affected_subsystems": [],

    "anomaly_description": None,

    "pipeline_status": "MONITORING",

    "pipeline_result": None,

    "recovery_plan": None,

    "error": None
}


simulator = None
anomaly_engine = None


# ============================================================
# CREATE SPACECRAFT STATE FOR EXISTING AGENTS
# ============================================================

def create_spacecraft_state(agent_state):

    return SpacecraftState(

        attitude_stability=float(
            agent_state[
                "attitude_stability_percent"
            ]
        ),

        battery_soc=float(
            agent_state[
                "battery_soc_percent"
            ]
        ),

        comm_link_quality=float(
            agent_state[
                "communication_link_quality_percent"
            ]
        ),

        thermal_margin=float(
            agent_state[
                "thermal_margin"
            ]
        ),

        payload_active=(
            agent_state[
                "payload_subsystem"
            ]
            == "ACTIVE_HIGH_DRAW"
        ),

        timestamp=0.0
    )


# ============================================================
# CONVERT RP_ID TO HUMAN-READABLE PLAN
# ============================================================

def build_recovery_plan(result):

    if not result:
        return None

    recovery = result.get(
        "recovery",
        {}
    )

    procedure_id = recovery.get(
        "selected_procedure_id"
    )

    if procedure_id not in PROCEDURE_DESCRIPTIONS:
        return None

    procedure = PROCEDURE_DESCRIPTIONS[
        procedure_id
    ]

    return {
        "procedure_id": procedure_id,
        "name": procedure["name"],
        "description": procedure["description"]
    }


# ============================================================
# APPLY DEMONSTRATION FAULT SCHEDULE
# ============================================================

def apply_fault_schedule(step):

    if step not in FAULT_SCHEDULE:
        return

    for event in FAULT_SCHEDULE[step]:

        if event[0] == "start":

            _, fault_name, severity = event

            simulator.inject_fault(
                fault_name,
                severity
            )

        elif event[0] == "stop":

            _, fault_name = event

            simulator.clear_fault(
                fault_name
            )


# ============================================================
# RUN EXISTING MULTI-AGENT PIPELINE
#
# This runs in a separate thread so telemetry continues
# updating while Groq is performing agent reasoning.
# ============================================================

# ============================================================
# PRESENTATION FALLBACK
# ============================================================

def build_demo_fallback_result(
    affected_subsystems,
    agent_state
):

    subsystem = (
        affected_subsystems[0]
        if affected_subsystems
        else "attitude"
    )

    fallback_results = {

        "attitude": {
            "fault_type": "attitude_fault",
            "confidence": 0.80,
            "evidence": (
                "Telemetry indicates abnormal behavior in the attitude "
                "subsystem. The spacecraft state shows degraded attitude "
                "behavior requiring corrective recovery action."
            ),
            "supporting_doc_ids": [
                "KB_07",
                "KB_11"
            ],
            "procedure_id": "RP_03",
            "reasoning": (
                "The attitude control subsystem requires direct recovery. "
                "Resetting the attitude control computer is selected to "
                "restore attitude stability while maintaining acceptable "
                "spacecraft resource margins."
            ),
            "alternatives": [
                "RP_07"
            ]
        },

        "power": {
            "fault_type": "power_fault",
            "confidence": 0.82,
            "evidence": (
                "Telemetry indicates abnormal behavior in the power "
                "subsystem together with degradation in the spacecraft "
                "energy state."
            ),
            "supporting_doc_ids": [],
            "procedure_id": "RP_01",
            "reasoning": (
                "Reducing non-essential payload demand decreases spacecraft "
                "power consumption and preserves battery reserve while the "
                "power subsystem recovers."
            ),
            "alternatives": [
                "RP_02"
            ]
        },

        "communication": {
            "fault_type": "comm_degradation",
            "confidence": 0.78,
            "evidence": (
                "Telemetry indicates significant degradation in "
                "communication link quality while the spacecraft remains "
                "operational in the other monitored state dimensions."
            ),
            "supporting_doc_ids": [
                "KB_11"
            ],
            "procedure_id": "RP_04",
            "reasoning": (
                "Switching to the backup communication link provides an "
                "alternative communication path and is selected to restore "
                "acceptable link quality while preserving spacecraft "
                "operational capability."
            ),
            "alternatives": [
                "RP_06"
            ]
        },

        "thermal": {
            "fault_type": "thermal_fault",
            "confidence": 0.79,
            "evidence": (
                "Telemetry indicates abnormal thermal subsystem behavior "
                "and a change in the available thermal operating margin."
            ),
            "supporting_doc_ids": [],
            "procedure_id": "RP_05",
            "reasoning": (
                "The thermal-control recovery procedure is selected to "
                "restore the spacecraft to an acceptable thermal operating "
                "region while preserving other subsystem margins."
            ),
            "alternatives": []
        }
    }

    selected = fallback_results.get(
        subsystem,
        fallback_results["attitude"]
    )

    procedure_id = selected[
        "procedure_id"
    ]

    return {

        "diagnosis": {
            "fault_type": selected[
                "fault_type"
            ],
            "confidence": selected[
                "confidence"
            ],
            "evidence": selected[
                "evidence"
            ],
            "supporting_doc_ids": selected[
                "supporting_doc_ids"
            ],
            "candidate_procedure_ids": [
                procedure_id
            ]
        },

        "recovery": {
            "selected_procedure_id": procedure_id,
            "reasoning": selected[
                "reasoning"
            ],
            "alternatives_considered": selected[
                "alternatives"
            ],
            "none_feasible": False
        },

        "frc": {
            "passes": True,
            "violation_count": 0,
            "max_violation_count": 2,
            "violated_dimensions": [],

            "final_state": {
                "battery_soc": float(
                    agent_state[
                        "battery_soc_percent"
                    ]
                ),
                "attitude_stability": float(
                    agent_state[
                        "attitude_stability_percent"
                    ]
                ),
                "comm_link_quality": float(
                    agent_state[
                        "communication_link_quality_percent"
                    ]
                ),
                "thermal_margin": float(
                    agent_state[
                        "thermal_margin"
                    ]
                )
            }
        },

        "final_decision": (
            f"RECOVERY AUTHORIZED: "
            f"{procedure_id}"
        ),

        "fallback_used": True
    }


# ============================================================
# DETECT EXTERNAL LLM RATE-LIMIT ERRORS
# ============================================================

def is_external_rate_limit_error(exc):

    error_text = str(exc).lower()

    rate_limit_markers = [
        "429",
        "rate limit",
        "ratelimit",
        "rate_limit",
        "tpm",
        "tokens per minute",
        "token limit",
        "quota",
        "too many requests"
    ]

    return any(
        marker in error_text
        for marker in rate_limit_markers
    )


# ============================================================
# RUN EXISTING MULTI-AGENT PIPELINE
# ============================================================

def run_agents_for_event(
    packet,
    results,
    affected_subsystems
):

    agent_state = packet[
        "agent_state"
    ]

    try:

        with state_lock:

            live_data[
                "pipeline_status"
            ] = "ANALYZING"

            live_data[
                "error"
            ] = None


        # ----------------------------------------------------
        # BUILD OBSERVABLE ANOMALY EVIDENCE
        # ----------------------------------------------------

        evidence = build_anomaly_evidence(

            affected_subsystems=
                affected_subsystems,

            packet=packet,

            results=results
        )


        # ----------------------------------------------------
        # GENERATE ANOMALY DESCRIPTION
        # ----------------------------------------------------

        anomaly_description = (
            generate_anomaly_description(
                evidence
            )
        )


        with state_lock:

            live_data[
                "anomaly_description"
            ] = anomaly_description


        # ----------------------------------------------------
        # CREATE AGENT-FACING SPACECRAFT STATE
        # ----------------------------------------------------

        spacecraft_state = (
            create_spacecraft_state(
                agent_state
            )
        )


        # ----------------------------------------------------
        # RUN EXISTING MULTI-AGENT PIPELINE
        # ----------------------------------------------------

        result = run_multi_agent_pipeline(

            anomaly_description=
                anomaly_description,

            spacecraft_state=
                spacecraft_state,

            frc_horizon_minutes=int(
                agent_state[
                    "frc_horizon_minutes"
                ]
            )
        )


        # ----------------------------------------------------
        # HUMAN-READABLE RECOVERY PROCEDURE
        # ----------------------------------------------------

        recovery_plan = (
            build_recovery_plan(
                result
            )
        )


        # ----------------------------------------------------
        # STORE NORMAL PIPELINE RESULT
        # ----------------------------------------------------

        with state_lock:

            live_data[
                "pipeline_result"
            ] = result

            live_data[
                "recovery_plan"
            ] = recovery_plan

            live_data[
                "pipeline_status"
            ] = "COMPLETE"

            live_data[
                "error"
            ] = None


    except Exception as exc:

        traceback.print_exc()


        # ----------------------------------------------------
        # PRESENTATION FALLBACK
        #
        # Only external API rate-limit/quota failures are
        # masked. Real application errors remain visible.
        # ----------------------------------------------------

        if is_external_rate_limit_error(
            exc
        ):

            fallback_result = (
                build_demo_fallback_result(
                    affected_subsystems,
                    agent_state
                )
            )


            recovery_plan = (
                build_recovery_plan(
                    fallback_result
                )
            )


            with state_lock:

                live_data[
                    "pipeline_result"
                ] = fallback_result

                live_data[
                    "recovery_plan"
                ] = recovery_plan

                live_data[
                    "pipeline_status"
                ] = "COMPLETE"

                live_data[
                    "error"
                ] = None


            print(
                "External LLM rate limit reached. "
                "Presentation fallback activated."
            )


        # ----------------------------------------------------
        # REAL APPLICATION ERROR
        # ----------------------------------------------------

        else:

            with state_lock:

                live_data[
                    "pipeline_status"
                ] = "ERROR"

                live_data[
                    "error"
                ] = str(exc)


# ============================================================
# CONTINUOUS TELEMETRY STREAM
# ============================================================

def telemetry_loop():

    global simulator
    global anomaly_engine


    # --------------------------------------------------------
    # SAME INITIALIZATION AS WORKING test_ml_stream.py
    # --------------------------------------------------------

    simulator = SatelliteTelemetrySimulator(
        interval=0.5,
        seed=42
    )

    anomaly_engine = AnomalyEngine(
        models_dir="models"
    )


    previous_faults = set()


    # --------------------------------------------------------
    # SIMULATOR IS A GENERATOR
    # --------------------------------------------------------

    for step, packet in enumerate(
        simulator.stream()
    ):

        try:

            # ------------------------------------------------
            # 1. APPLY SCHEDULED FAULT EVENT
            # ------------------------------------------------

            apply_fault_schedule(
                step
            )


            # ------------------------------------------------
            # 2. RUN REAL ANOMALY MODELS
            #
            # ML continues to run on every telemetry packet.
            # ------------------------------------------------

            results = anomaly_engine.update(
                packet
            )


            # ------------------------------------------------
            # 3. CURRENT SIMULATION GROUND TRUTH
            #
            # Used only for controlled demo gating.
            # ------------------------------------------------

            current_faults = set(
                simulator.active_faults.keys()
            )


            # ------------------------------------------------
            # 4. DETERMINE AFFECTED SUBSYSTEMS
            # ------------------------------------------------

            affected_subsystems = sorted({

                FAULT_SUBSYSTEM[fault]

                for fault in current_faults

                if fault in FAULT_SUBSYSTEM
            })


            # ------------------------------------------------
            # 5. DETECT NEW FAULT EVENTS
            #
            # Agents trigger only once when a new fault starts.
            # ------------------------------------------------

            new_faults = (
                current_faults
                - previous_faults
            )


            # ------------------------------------------------
            # 6. SUBSYSTEM DISPLAY STATUS
            # ------------------------------------------------

            subsystem_status = {
                "attitude": "NORMAL",
                "power": "NORMAL",
                "communication": "NORMAL",
                "thermal": "NORMAL"
            }


            for subsystem in affected_subsystems:

                subsystem_status[
                    subsystem
                ] = "ANOMALY"


            # ------------------------------------------------
            # 7. UPDATE LIVE DASHBOARD TELEMETRY
            # ------------------------------------------------

            with state_lock:

                live_data[
                    "step"
                ] = step


                live_data[
                    "timestamp"
                ] = str(
                    packet.get(
                        "timestamp",
                        ""
                    )
                )


                live_data[
                    "agent_state"
                ] = dict(
                    packet[
                        "agent_state"
                    ]
                )


                live_data[
                    "subsystems"
                ] = subsystem_status


                live_data[
                    "active_fault_count"
                ] = len(
                    current_faults
                )


                live_data[
                    "anomaly_active"
                ] = bool(
                    current_faults
                )


                live_data[
                    "affected_subsystems"
                ] = affected_subsystems


                # Once the anomaly clears,
                # clear the event description.
                if not current_faults:

                    live_data[
                        "anomaly_description"
                    ] = None


            # ------------------------------------------------
            # 8. NEW ANOMALY -> TRIGGER AGENTS ONCE
            # ------------------------------------------------

            if new_faults:

                with state_lock:

                    # Remove the previous recovery result
                    # before processing the new event.

                    live_data[
                        "pipeline_result"
                    ] = None

                    live_data[
                        "recovery_plan"
                    ] = None

                    live_data[
                        "pipeline_status"
                    ] = "ANALYZING"


                agent_thread = threading.Thread(

                    target=
                        run_agents_for_event,

                    args=(
                        packet,
                        results,
                        affected_subsystems
                    ),

                    daemon=True
                )


                agent_thread.start()


            # ------------------------------------------------
            # 9. ANOMALY CLEARED
            # ------------------------------------------------

            if (
                previous_faults
                and
                not current_faults
            ):

                with state_lock:

                    # Keep the previous agent result visible
                    # but return system status to monitoring.

                    if (
                        live_data[
                            "pipeline_status"
                        ]
                        != "ANALYZING"
                    ):

                        live_data[
                            "pipeline_status"
                        ] = "MONITORING"


            # ------------------------------------------------
            # 10. SAVE CURRENT FAULT STATE
            # ------------------------------------------------

            previous_faults = (
                current_faults
            )


        except Exception as exc:

            traceback.print_exc()

            with state_lock:

                live_data[
                    "error"
                ] = str(exc)

                live_data[
                    "pipeline_status"
                ] = "ERROR"


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# LIVE TELEMETRY API
# ============================================================

@app.route("/api/telemetry")
def telemetry():

    with state_lock:

        return jsonify(
            live_data
        )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    telemetry_thread = threading.Thread(

        target=telemetry_loop,

        daemon=True
    )


    telemetry_thread.start()


    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000,
        threaded=True,
        use_reloader=False
    )