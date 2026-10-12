import json
from typing import Dict, Any, List

from recovery_procedure import PROCEDURE_LIBRARY
from spacecraft_state import SpacecraftState
from state_prediction import predict_trajectory
from frc_evaluator import evaluate_procedure

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found. Check your .env file."
    )

client = Groq(api_key=api_key)
# =========================================================
# TOOL 1 — Inspect spacecraft state
# =========================================================

def inspect_spacecraft_state(state: SpacecraftState) -> dict:

    return {
        "battery_soc": state.battery_soc,
        "thermal_margin": state.thermal_margin,
        "attitude_stability": state.attitude_stability,
        "comm_link_quality": state.comm_link_quality,
        "payload_active": state.payload_active,
        "timestamp": state.timestamp
    }


# =========================================================
# TOOL 2 — Get candidate procedures
# =========================================================

def get_candidate_procedures(
    fault_type: str,
    excluded_ids: List[str]
) -> list:

    candidates = []

    for procedure in PROCEDURE_LIBRARY:

        if procedure.fault != fault_type:
            continue

        if procedure.procedure_id in excluded_ids:
            continue

        candidates.append({
            "procedure_id": procedure.procedure_id,
            "fault": procedure.fault,
            "actions": procedure.actions,
            "expected_effects": procedure.expected_effects,
            "preconditions": procedure.preconditions
        })

    return candidates


# =========================================================
# TOOL 3 — Check procedure preconditions
# =========================================================

def check_preconditions(
    procedure_id: str,
    state: SpacecraftState
) -> dict:

    procedure = next(
        (
            p for p in PROCEDURE_LIBRARY
            if p.procedure_id == procedure_id
        ),
        None
    )

    if procedure is None:
        return {
            "procedure_id": procedure_id,
            "feasible": False,
            "reason": "Unknown procedure"
        }

    failures = []

    for parameter, required_value in procedure.preconditions.items():

        actual_value = getattr(
            state,
            parameter,
            None
        )

        if actual_value is None:
            failures.append(
                f"{parameter} unavailable"
            )
            continue

        if actual_value < required_value:
            failures.append(
                f"{parameter}={actual_value} "
                f"< required {required_value}"
            )

    return {
        "procedure_id": procedure_id,
        "feasible": len(failures) == 0,
        "failed_preconditions": failures
    }


# =========================================================
# TOOL 4 — Simulate procedure
# =========================================================

def simulate_procedure(
    procedure_id: str,
    state: SpacecraftState,
    horizon_minutes: int = 30
) -> dict:

    procedure = next(
        (
            p for p in PROCEDURE_LIBRARY
            if p.procedure_id == procedure_id
        ),
        None
    )

    if procedure is None:
        return {
            "error": "Unknown procedure"
        }

    trajectory = predict_trajectory(
        state,
        procedure,
        horizon_minutes=horizon_minutes
    )

    return {
        "procedure_id": procedure_id,
        "trajectory": trajectory
    }


# =========================================================
# TOOL 5 — Evaluate FRC
# =========================================================

def evaluate_recovery_frc(
    procedure_id: str,
    trajectory: list
) -> dict:

    result = evaluate_procedure(
        trajectory,
        PROCEDURE_LIBRARY,
        threshold=2
    )

    if result is None:
        return {
            "passes": False,
            "reason": "FRC evaluator returned no result"
        }

    result["procedure_id"] = procedure_id

    return result


# =========================================================
# TOOL REGISTRY
# =========================================================

TOOLS = {
    "inspect_spacecraft_state": inspect_spacecraft_state,
    "get_candidate_procedures": get_candidate_procedures,
    "check_preconditions": check_preconditions,
    "simulate_procedure": simulate_procedure,
    "evaluate_recovery_frc": evaluate_recovery_frc
}
from groq import Groq


client = Groq()


SYSTEM_PROMPT = """
You are the Recovery Planning Agent for an autonomous spacecraft
recovery system.

Your responsibility is to select a safe recovery procedure for a
diagnosed spacecraft fault.

You operate using tools.

You MUST NOT simply guess a procedure.

You should:

1. Inspect the spacecraft state.
2. Obtain candidate procedures for the diagnosed fault.
3. Check preconditions of candidate procedures.
4. Reject procedures whose preconditions are not satisfied.
5. Simulate feasible procedures.
6. Evaluate their Future Recovery Capability (FRC).
7. If a procedure fails FRC, do not authorize it.
8. Consider another feasible procedure.
9. Continue until:
   - a procedure passes FRC, or
   - no feasible procedure remains.

The FRC evaluator is the safety authority.
Never override a failed safety check.

Return a final JSON object with:

{
    "selected_procedure_id": "... or null",
    "reasoning": "...",
    "alternatives_considered": [...],
    "rejected_procedures": [...],
    "none_feasible": true/false
}
"""


def run_recovery_agent(
    diagnosis: dict,
    spacecraft_state: SpacecraftState,
    excluded_procedure_ids: list | None = None,
    max_steps: int = 12
) -> dict:

    if excluded_procedure_ids is None:
        excluded_procedure_ids = []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": json.dumps({
                "diagnosis": diagnosis,
                "spacecraft_state": inspect_spacecraft_state(
                    spacecraft_state
                ),
                "already_rejected": excluded_procedure_ids
            })
        }
    ]

    rejected = list(excluded_procedure_ids)
    alternatives = []

    for step in range(max_steps):

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "inspect_spacecraft_state",
                        "description": "Inspect current spacecraft state.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_candidate_procedures",
                        "description": "Retrieve procedures applicable to a diagnosed fault.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "fault_type": {
                                    "type": "string"
                                },
                                "excluded_ids": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            },
                            "required": [
                                "fault_type",
                                "excluded_ids"
                            ]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "check_preconditions",
                        "description": "Check whether a procedure can currently be executed.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "procedure_id": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "procedure_id"
                            ]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "simulate_procedure",
                        "description": "Simulate spacecraft trajectory after applying a procedure.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "procedure_id": {
                                    "type": "string"
                                },
                                "horizon_minutes": {
                                    "type": "integer"
                                }
                            },
                            "required": [
                                "procedure_id"
                            ]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "evaluate_recovery_frc",
                        "description": "Evaluate future recovery capability of a simulated trajectory.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "procedure_id": {
                                    "type": "string"
                                },
                                "trajectory": {
                                    "type": "array"
                                }
                            },
                            "required": [
                                "procedure_id",
                                "trajectory"
                            ]
                        }
                    }
                }
            ]
        )

        message = response.choices[0].message

        # -------------------------------------------------
        # Agent has finished reasoning
        # -------------------------------------------------

        if not message.tool_calls:

            content = message.content

            try:
                return json.loads(content)

            except json.JSONDecodeError:

                return {
                    "selected_procedure_id": None,
                    "reasoning": content,
                    "alternatives_considered": alternatives,
                    "rejected_procedures": rejected,
                    "none_feasible": True
                }

        # -------------------------------------------------
        # Execute tools requested by the agent
        # -------------------------------------------------

        messages.append(message)

        for tool_call in message.tool_calls:

            name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print(
                f"\n[Recovery Agent] TOOL → {name}"
            )

            print(
                json.dumps(
                    arguments,
                    indent=2
                )
            )

            # ---------------------------------------------
            # Execute actual Python tool
            # ---------------------------------------------

            if name == "inspect_spacecraft_state":

                result = inspect_spacecraft_state(
                    spacecraft_state
                )

            elif name == "get_candidate_procedures":

                result = get_candidate_procedures(
                    arguments["fault_type"],
                    arguments["excluded_ids"]
                )

                for candidate in result:
                    if candidate["procedure_id"] not in alternatives:
                        alternatives.append(
                            candidate["procedure_id"]
                        )

            elif name == "check_preconditions":

                result = check_preconditions(
                    arguments["procedure_id"],
                    spacecraft_state
                )

                if not result["feasible"]:

                    if arguments["procedure_id"] not in rejected:
                        rejected.append(
                            arguments["procedure_id"]
                        )

            elif name == "simulate_procedure":

                result = simulate_procedure(
                    arguments["procedure_id"],
                    spacecraft_state,
                    arguments.get(
                        "horizon_minutes",
                        30
                    )
                )

            elif name == "evaluate_recovery_frc":

                result = evaluate_recovery_frc(
                    arguments["procedure_id"],
                    arguments["trajectory"]
                )

                if not result["passes"]:

                    if arguments["procedure_id"] not in rejected:
                        rejected.append(
                            arguments["procedure_id"]
                        )

            else:

                result = {
                    "error": f"Unknown tool: {name}"
                }

            # ---------------------------------------------
            # Give tool observation back to the agent
            # ---------------------------------------------

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    # -----------------------------------------------------
    # Safety fallback if agent exceeds step limit
    # -----------------------------------------------------

    return {
        "selected_procedure_id": None,
        "reasoning": (
            "Recovery agent exceeded maximum reasoning "
            "steps without reaching a safe decision."
        ),
        "alternatives_considered": alternatives,
        "rejected_procedures": rejected,
        "none_feasible": True
    }