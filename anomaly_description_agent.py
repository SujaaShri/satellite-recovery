def build_anomaly_evidence(
    affected_subsystems,
    packet,
    results
):

    state = packet["agent_state"]

    evidence = {
        "affected_subsystems": list(
            affected_subsystems
        ),

        "spacecraft_state": {
            "attitude_stability_percent":
                state["attitude_stability_percent"],

            "battery_soc_percent":
                state["battery_soc_percent"],

            "communication_link_quality_percent":
                state[
                    "communication_link_quality_percent"
                ],

            "thermal_margin":
                state["thermal_margin"],

            "payload_subsystem":
                state["payload_subsystem"],

            "frc_horizon_minutes":
                state["frc_horizon_minutes"]
        },

        "detector_scores": {}
    }

    for subsystem in affected_subsystems:

        result = results.get(
            subsystem
        )

        if (
            result
            and result.get("ready")
        ):

            evidence[
                "detector_scores"
            ][subsystem] = result.get(
                "score"
            )

    return evidence


def generate_anomaly_description(
    evidence
):

    affected = evidence[
        "affected_subsystems"
    ]

    state = evidence[
        "spacecraft_state"
    ]

    subsystem_text = ", ".join(
        affected
    )

    description = (
        f"Anomalous telemetry behavior has been detected "
        f"in the {subsystem_text} subsystem. "
        f"Current spacecraft state: "
        f"attitude stability "
        f"{state['attitude_stability_percent']:.1f}%, "
        f"battery SOC "
        f"{state['battery_soc_percent']:.1f}%, "
        f"communication link quality "
        f"{state['communication_link_quality_percent']:.1f}%, "
        f"thermal margin "
        f"{state['thermal_margin']:.1f}, "
        f"payload state "
        f"{state['payload_subsystem']}. "
        f"The anomaly should be diagnosed using the "
        f"observed spacecraft state and subsystem behavior."
    )

    return description