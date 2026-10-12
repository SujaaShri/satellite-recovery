from telemetry_simulator import SatelliteTelemetrySimulator


sim = SatelliteTelemetrySimulator(interval=0.5)


# Scenario schedule for demonstration.
# These events happen while the stream itself continues forever.
scenario_schedule = {

    # Attitude fault
    20: [
        ("start", "spin_drift", "moderate")
    ],

    40: [
        ("stop", "spin_drift")
    ],

    # Communication fault
    60: [
        ("start", "signal_degradation", "moderate")
    ],

    80: [
        ("stop", "signal_degradation")
    ],

    # Power fault
    100: [
        ("start", "voltage_degradation", "moderate")
    ],

    120: [
        ("stop", "voltage_degradation")
    ],

    # Thermal fault
    140: [
        ("start", "component_overheating", "moderate")
    ],

    160: [
        ("stop", "component_overheating")
    ],

    # Power + Thermal together
    180: [
        ("start", "voltage_degradation", "moderate"),
        ("start", "multi_component_heating", "moderate")
    ],

    210: [
        ("stop", "voltage_degradation"),
        ("stop", "multi_component_heating")
    ],

    # Attitude + Communication + Power together
    230: [
        ("start", "excessive_spin", "moderate"),
        ("start", "signal_degradation", "moderate"),
        ("start", "coupled_power_stress", "mild")
    ],

    270: [
        ("stop", "excessive_spin"),
        ("stop", "signal_degradation"),
        ("stop", "coupled_power_stress")
    ]
}


try:

    for i, packet in enumerate(sim.stream()):

        # -----------------------------------------
        # Execute scheduled fault events
        # -----------------------------------------

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

                    sim.clear_fault(fault)

        # -----------------------------------------
        # Agent-facing spacecraft state
        # -----------------------------------------

        state = packet["agent_state"]

        # Debug only
        active_faults = (
            ", ".join(sim.active_faults.keys())
            if sim.active_faults
            else "NONE"
        )

        print(
            f'{packet["timestamp"]} '
            f'| ATT: {state["attitude_stability_percent"]}% '
            f'| COMM: {state["communication_link_quality_percent"]}% '
            f'| SOC: {state["battery_soc_percent"]}% '
            f'| THERM: {state["thermal_margin"]} '
            f'| PAYLOAD: {state["payload_subsystem"]} '
            f'| FRC: {state["frc_horizon_minutes"]} min '
            f'| SIM_FAULT: {active_faults}'
        )


except KeyboardInterrupt:

    print("\nTelemetry stream stopped by user.")