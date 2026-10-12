import os, json
from groq import Groq
from dotenv import load_dotenv
from recovery_procedure import PROCEDURE_LIBRARY, check_preconditions

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL="openai/gpt-oss-20b"

def get_feasibility_report(state, fault_type, excluded_ids=None):
    excluded_ids = excluded_ids or []
    candidates = [p for p in PROCEDURE_LIBRARY if p.fault == fault_type and p.procedure_id not in excluded_ids]
    return [{"procedure_id": p.procedure_id, "actions": p.actions, "expected_effects": p.expected_effects,
              "preconditions": p.preconditions, "feasible": check_preconditions(state, p)} for p in candidates]

PLANNING_SYSTEM_PROMPT = """You are a spacecraft recovery planning agent. Given a diagnosis
and a feasibility report of every remaining candidate procedure, select the best FEASIBLE one.
Prefer the one whose expected_effects most directly address the fault. If none feasible, say so.
Respond ONLY with JSON:
{"selected_procedure_id": "... or null", "reasoning": "...",
 "alternatives_considered": ["..."], "none_feasible": false}"""

def plan_recovery(diagnosis, state, excluded_procedure_ids=None):
    fault_type = diagnosis["fault_type"]
    excluded_procedure_ids = excluded_procedure_ids or []
    feasibility_report = get_feasibility_report(state, fault_type, excluded_procedure_ids)

    if not feasibility_report or not any(r["feasible"] for r in feasibility_report):
        return {"selected_procedure": None, "reasoning": "No feasible, non-excluded procedure remains.",
                "feasibility_report": feasibility_report}

    user_prompt = (f"Diagnosis:\n{json.dumps(diagnosis, indent=2)}\n\n"
                    f"Excluded IDs: {excluded_procedure_ids}\n\n"
                    f"Feasibility report:\n{json.dumps(feasibility_report, indent=2)}\n\nSelect the best procedure.")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                   {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    decision = json.loads(response.choices[0].message.content)

    if decision.get("none_feasible") or not decision.get("selected_procedure_id"):
        return {"selected_procedure": None, "reasoning": decision.get("reasoning", ""),
                "feasibility_report": feasibility_report}

    selected = next((p for p in PROCEDURE_LIBRARY if p.procedure_id == decision["selected_procedure_id"]), None)
    if not selected:
        raise ValueError(f"Agent selected unknown procedure_id: {decision['selected_procedure_id']}")

    matching = next((r for r in feasibility_report if r["procedure_id"] == selected.procedure_id), None)
    if matching is None or not matching["feasible"]:
        raise ValueError(f"Agent selected {selected.procedure_id} but it's infeasible/excluded.")

    return {"selected_procedure": selected.to_dict(), "reasoning": decision.get("reasoning", ""),
            "alternatives_considered": decision.get("alternatives_considered", []),
            "feasibility_report": feasibility_report}