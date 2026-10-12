import os, json
from groq import Groq
from dotenv import load_dotenv
from rag_retriever import FaultRetriever

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
retriever = FaultRetriever()
MODEL = "openai/gpt-oss-120b"  # verify still live: https://console.groq.com/docs/models

DIAGNOSIS_SYSTEM_PROMPT = """You are a spacecraft fault diagnosis agent. Given an observed
telemetry anomaly and retrieved fault-case documents, determine the most likely fault type
from EXACTLY one of: battery_low, thermal_excursion, attitude_fault, comm_degradation.
Base your diagnosis on the retrieved documents and cite which doc(s) support it.
Respond ONLY with JSON:
{"fault_type": "...", "evidence": "...", "confidence": 0.0,
 "supporting_doc_ids": ["..."], "candidate_procedure_ids": ["..."]}"""

def diagnose(anomaly_description: str, top_k_docs: int = 3) -> dict:
    retrieved_docs = retriever.retrieve(anomaly_description, top_k=top_k_docs)
    docs_text = "\n\n".join(f"[{d['doc_id']}] {d['title']}\n{d['content']}" for d in retrieved_docs)
    user_prompt = f"Observed anomaly:\n{anomaly_description}\n\nRetrieved documents:\n{docs_text}\n\nDiagnose as JSON."
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
                   {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    result = json.loads(response.choices[0].message.content)
    result["retrieved_doc_ids"] = [d["doc_id"] for d in retrieved_docs]
    return result