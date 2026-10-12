import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from spacecraft_state import SpacecraftState
from recovery_procedure import PROCEDURE_LIBRARY

# Import your existing RAG retrieval function here.
# CHANGE THIS IMPORT ONLY if your existing function has another name.
from rag_retriever import FaultRetriever


retriever = FaultRetriever()


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found. Check your .env file."
    )

client = Groq(api_key=api_key)


# =========================================================
# TOOLS
# =========================================================

def inspect_spacecraft_state(
    spacecraft_state: SpacecraftState
) -> dict:

    return spacecraft_state.as_dict()


def search_knowledge_base(
    query: str,
    top_k: int = 5
) -> list:

    documents = retriever.retrieve_documents(
        query=query,
        top_k=top_k
    )

    results = []

    for doc in documents:
        if isinstance(doc, dict):
            results.append(doc)
        else:
            results.append({
                "content": str(doc)
            })

    return results


def get_fault_procedures(
    fault_type: str
) -> list:

    procedures = []

    for procedure in PROCEDURE_LIBRARY:

        if procedure.fault == fault_type:

            procedures.append({
                "procedure_id": procedure.procedure_id,
                "fault": procedure.fault,
                "actions": procedure.actions,
                "expected_effects": procedure.expected_effects,
                "preconditions": procedure.preconditions
            })

    return procedures


TOOLS = {
    "inspect_spacecraft_state": inspect_spacecraft_state,
    "search_knowledge_base": search_knowledge_base,
    "get_fault_procedures": get_fault_procedures
}


# =========================================================
# TOOL DEFINITIONS FOR GROQ
# =========================================================

TOOL_SCHEMAS = [

    {
        "type": "function",
        "function": {
            "name": "inspect_spacecraft_state",
            "description": (
                "Inspect the current spacecraft telemetry/state. "
                "Use this to understand power, thermal, attitude, "
                "communication and payload conditions."
            ),
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
            "name": "search_knowledge_base",
            "description": (
                "Search the spacecraft fault knowledge base for "
                "diagnostic evidence, known fault signatures, "
                "and relevant operational documentation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Diagnostic query describing the observed "
                            "spacecraft behavior."
                        )
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of documents to retrieve."
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_fault_procedures",
            "description": (
                "Retrieve recovery procedures associated with a "
                "candidate fault type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fault_type": {
                        "type": "string"
                    }
                },
                "required": ["fault_type"]
            }
        }
    }
]


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are the Diagnosis Agent in an autonomous spacecraft
fault diagnosis and recovery system.

You are NOT the recovery planner.

Your job is to investigate an observed spacecraft anomaly
and determine the most likely fault.

You have access to tools.

You should reason through the available evidence and use
tools when necessary.

Typical workflow:

1. Inspect spacecraft state.
2. Search the knowledge base for the observed behavior.
3. Compare retrieved fault signatures with telemetry.
4. If necessary, perform additional KB searches.
5. Determine the most likely fault.
6. Retrieve candidate recovery procedures associated with
   the suspected fault.
7. Return a structured diagnosis.

Do NOT invent evidence.

Do NOT invent fault types.

Do NOT invent procedure IDs.

The retrieved knowledge base must support the diagnosis.

Return JSON only in the final response:

{
    "fault_type": "...",
    "evidence": "...",
    "confidence": 0.0,
    "supporting_doc_ids": [],
    "candidate_procedure_ids": [],
    "retrieved_doc_ids": []
}
"""


# =========================================================
# AGENT
# =========================================================

def run_diagnosis_agent(
    anomaly_description: str,
    spacecraft_state: SpacecraftState,
    max_iterations: int = 8
) -> dict:

    print("\n" + "=" * 70)
    print("DIAGNOSIS ReAct AGENT")
    print("=" * 70)

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": (
                "Observed anomaly:\n"
                f"{anomaly_description}\n\n"
                "Investigate this anomaly and determine the most "
                "likely spacecraft fault."
            )
        }
    ]

    for iteration in range(max_iterations):

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0
        )

        message = response.choices[0].message

        # -------------------------------------------------
        # Agent wants to use tools
        # -------------------------------------------------

        if message.tool_calls:

            messages.append(message)

            for tool_call in message.tool_calls:

                name = tool_call.function.name

                arguments = json.loads(
                    tool_call.function.arguments
                )

                print(
                    f"\n[Diagnosis Agent] TOOL → {name}"
                )

                print(
                    json.dumps(
                        arguments,
                        indent=2
                    )
                )

                if name == "inspect_spacecraft_state":

                    result = inspect_spacecraft_state(
                        spacecraft_state
                    )

                elif name == "search_knowledge_base":

                    result = search_knowledge_base(
                        **arguments
                    )

                elif name == "get_fault_procedures":

                    result = get_fault_procedures(
                        **arguments
                    )

                else:

                    result = {
                        "error": f"Unknown tool: {name}"
                    }

                print(
                    "[Diagnosis Agent] TOOL RESULT"
                )

                print(
                    json.dumps(
                        result,
                        indent=2,
                        default=str
                    )
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        default=str
                    )
                })

            continue

        # -------------------------------------------------
        # Agent has reached a final diagnosis
        # -------------------------------------------------

        final_text = message.content.strip()

        print("\n[Diagnosis Agent] FINAL RESPONSE")
        print(final_text)

        try:

            result = json.loads(final_text)

        except json.JSONDecodeError:

            # Handle markdown JSON if the model wraps it.
            cleaned = final_text

            if "```json" in cleaned:
                cleaned = (
                    cleaned
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            result = json.loads(cleaned)

        required_fields = [
            "fault_type",
            "evidence",
            "confidence",
            "supporting_doc_ids",
            "candidate_procedure_ids",
            "retrieved_doc_ids"
        ]

        missing = [
            field
            for field in required_fields
            if field not in result
        ]

        if missing:
            raise ValueError(
                "Diagnosis agent response missing fields: "
                + ", ".join(missing)
            )

        return result

    raise RuntimeError(
        "Diagnosis agent exceeded maximum tool iterations."
    )