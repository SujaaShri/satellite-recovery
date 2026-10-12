# experiments.py

import json
import csv
import os

from orchestrator import app
from spacecraft_state import SpacecraftState


# =========================================================
# EXPERIMENT SCENARIOS
# =========================================================

SCENARIOS = [

    {
        "name": "Battery depletion during eclipse",

        "anomaly": (
            "Battery state of charge dropped from 80% to 60% "
            "while the payload remained active during an eclipse period."
        ),

        "state": SpacecraftState(
            battery_soc=80.0,
            thermal_margin=15.0,
            attitude_stability=95.0,
            comm_link_quality=90.0,
            payload_active=True,
            timestamp=0.0
        )
    },
    {
    "name": "Attitude fault with marginal battery",

    "anomaly": (
        "Attitude stability dropped from 95% to 60% "
        "while battery state of charge is marginal at 22%."
    ),

    "state": SpacecraftState(
        battery_soc=22.0,
        thermal_margin=15.0,
        attitude_stability=60.0,
        comm_link_quality=90.0,
        payload_active=True,
        timestamp=0.0
    )
},

    {
        "name": "Attitude control fault",

        "anomaly": (
            "Attitude stability suddenly dropped from 95% to 65% "
            "while battery state of charge remained healthy."
        ),

        "state": SpacecraftState(
            battery_soc=80.0,
            thermal_margin=15.0,
            attitude_stability=65.0,
            comm_link_quality=90.0,
            payload_active=True,
            timestamp=0.0
        )
    },


    {
        "name": "Thermal excursion",

        "anomaly": (
            "Thermal margin decreased rapidly from 15% to 7% "
            "during an eclipse period."
        ),

        "state": SpacecraftState(
            battery_soc=80.0,
            thermal_margin=7.0,
            attitude_stability=95.0,
            comm_link_quality=90.0,
            payload_active=True,
            timestamp=0.0
        )
    },


    {
        "name": "Communication degradation",

        "anomaly": (
            "Communication link quality dropped from 90% to 40% "
            "while battery state of charge and attitude stability "
            "remained nominal."
        ),

        "state": SpacecraftState(
            battery_soc=80.0,
            thermal_margin=15.0,
            attitude_stability=95.0,
            comm_link_quality=40.0,
            payload_active=True,
            timestamp=0.0
        )
    },


    {
        "name": "Attitude fault with critically low battery",

        "anomaly": (
            "Attitude stability dropped to 60% while battery "
            "state of charge is critically low at 5%."
        ),

        "state": SpacecraftState(
            battery_soc=5.0,
            thermal_margin=15.0,
            attitude_stability=60.0,
            comm_link_quality=90.0,
            payload_active=True,
            timestamp=0.0
        )
    }

]


# =========================================================
# RUN ONE EXPERIMENT
# =========================================================

def run_experiment(scenario):

    print("\n")
    print("=" * 70)
    print("SCENARIO:", scenario["name"])
    print("=" * 70)

    print("\nOBSERVED ANOMALY")
    print(scenario["anomaly"])

    print("\nINITIAL STATE")

    print(
        json.dumps(
            scenario["state"].as_dict(),
            indent=2
        )
    )

    initial_state = scenario["state"]

    # -----------------------------------------------------
    # LangGraph execution
    # -----------------------------------------------------

    result = app.invoke({

        "anomaly_description":
            scenario["anomaly"],

        "spacecraft_state":
            initial_state,

        "diagnosis":
            None,

        "plan":
            None,

        "frc_result":
            None,

        "rejected_procedure_ids":
            [],

        "replan_count":
            0,

        "max_replans":
            3,

        "final_decision":
            None
    })


    # -----------------------------------------------------
    # Extract results
    # -----------------------------------------------------

    diagnosis = result.get("diagnosis", {})

    plan = result.get("plan", {})

    frc = result.get("frc_result", {})

    selected = plan.get(
        "selected_procedure"
    )

    procedure_id = (
        selected["procedure_id"]
        if selected
        else None
    )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = {

        "scenario":
            scenario["name"],

        "diagnosed_fault":
            diagnosis.get("fault_type"),

        "diagnosis_confidence":
            diagnosis.get("confidence"),

        "retrieved_docs":
            ", ".join(
                diagnosis.get(
                    "retrieved_doc_ids",
                    []
                )
            ),

        "selected_procedure":
            procedure_id,

        "feasible":
            (
                any(
                    x["procedure_id"] == procedure_id
                    and x["feasible"]
                    for x in plan.get(
                        "feasibility_report",
                        []
                    )
                )
                if procedure_id
                else False
            ),

        "frc_pass":
            frc.get("passes", False),

        "replan_count":
            result.get(
                "replan_count",
                0
            ),

        "rejected_procedures":
            ", ".join(
                result.get(
                    "rejected_procedure_ids",
                    []
                )
            ),

        "final_decision":
            result.get(
                "final_decision"
            )
    }


    print("\n")
    print("-" * 70)
    print("EXPERIMENT SUMMARY")
    print("-" * 70)

    print(
        json.dumps(
            summary,
            indent=2
        )
    )

    return result, summary


# =========================================================
# RUN ALL EXPERIMENTS
# =========================================================

def main():

    results = []

    for scenario in SCENARIOS:

        result, summary = run_experiment(
            scenario
        )

        results.append(summary)


    # =====================================================
    # SAVE CSV
    # =====================================================

    output_file = "experiment_results.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(results)


    # =====================================================
    # FINAL TABLE
    # =====================================================

    print("\n")
    print("=" * 100)
    print("FINAL EXPERIMENT RESULTS")
    print("=" * 100)

    headers = [
        "scenario",
        "diagnosed_fault",
        "selected_procedure",
        "frc_pass",
        "replan_count",
        "final_decision"
    ]

    print(
        " | ".join(
            f"{h:25}"
            for h in headers
        )
    )

    print("-" * 100)

    for row in results:

        print(
            " | ".join(
                f"{str(row[h]):25}"
                for h in headers
            )
        )


    print("\nResults saved to:")
    print(output_file)


if __name__ == "__main__":
    main()