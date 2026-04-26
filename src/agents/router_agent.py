"""
Router Agent — Chief Medical Officer

Receives a patient case (symptoms + known conditions) and decides which
specialist agents to activate. Acts as the top-level dispatcher in the
LangGraph hospital graph.

Routing logic (in priority order):
  1. Direct condition → specialist match (CONDITION_SPECIALIST_MAP)
  2. Symptom → specialist match (SYMPTOM_SPECIALIST_MAP)
  3. Urgency escalation — critical cases always add Anesthesiology
  4. Minimum 1 specialist guaranteed (falls back to General Medicine)
"""

import requests
from .hospital_state import HospitalState
from .specialists.registry import (
    SPECIALISTS,
    SYMPTOM_SPECIALIST_MAP,
    CONDITION_SPECIALIST_MAP,
)

BACKEND_URL = "http://localhost:8000"


# Patient intake — fetch DB record and enrich state

def patient_intake(state: HospitalState) -> dict:
    """
    Fetch the patient record from the backend DB and populate:
      patient_name, age, conditions (from comorbidities), urgency.
    """
    patient_id = state["patient_id"]
    updates: dict = {"step_count": 1}

    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"sql": f"SELECT * FROM patients WHERE patient_id = '{patient_id}'"},
            timeout=5,
        )
        row = resp.json().get("rows", [{}])[0] if resp.ok else {}
        updates["patient_name"] = row.get("name", "Unknown")
        updates["age"] = row.get("age", 0)
        updates["patient_db_record"] = row
        updates["reasoning"] = [f"Intake: loaded patient {patient_id} — {row.get('name', 'Unknown')}"]

        # Pull comorbidities into conditions list
        comorbidity_resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"sql": f"SELECT condition FROM comorbidities WHERE patient_id = '{patient_id}'"},
            timeout=5,
        )
        if comorbidity_resp.ok:
            rows = comorbidity_resp.json().get("rows", [])
            db_conditions = [r["condition"] for r in rows if r.get("condition")]
            existing = state.get("conditions") or []
            merged = list(dict.fromkeys(existing + db_conditions))  # deduplicate, preserve order
            updates["conditions"] = merged
            if db_conditions:
                updates["reasoning"] = updates["reasoning"] + [
                    f"Intake: found comorbidities {db_conditions}"
                ]

    except Exception as exc:
        updates["reasoning"] = [f"Intake: DB unavailable ({exc}), proceeding with provided symptoms"]

    return updates


# Core routing logic

def _specialists_for_symptoms(symptoms: list[str]) -> list[str]:
    matched: list[str] = []
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        for key, specs in SYMPTOM_SPECIALIST_MAP.items():
            if key in symptom_lower:
                matched.extend(specs)
    return matched


def _specialists_for_conditions(conditions: list[str]) -> list[str]:
    matched: list[str] = []
    for condition in conditions:
        condition_lower = condition.lower().replace(" ", "_")
        if condition_lower in CONDITION_SPECIALIST_MAP:
            matched.append(CONDITION_SPECIALIST_MAP[condition_lower])
        else:
            # Fuzzy: check if condition is in any specialist's conditions list
            for spec_name, spec_data in SPECIALISTS.items():
                if any(condition_lower in c for c in spec_data["conditions"]):
                    matched.append(spec_name)
    return matched


def triage_router(state: HospitalState) -> dict:
    """
    Determine which specialists to activate for this patient.
    Returns updated state with active_specialists populated.
    """
    symptoms = state.get("symptoms") or []
    conditions = state.get("conditions") or []
    urgency = state.get("urgency", "medium")

    selected: list[str] = []

    # Priority 1: direct condition matches
    selected.extend(_specialists_for_conditions(conditions))

    # Priority 2: symptom matches
    selected.extend(_specialists_for_symptoms(symptoms))

    # Deduplicate while preserving priority order
    seen: set[str] = set()
    unique: list[str] = []
    for s in selected:
        if s not in seen and s in SPECIALISTS:
            seen.add(s)
            unique.append(s)

    # Priority 3: critical urgency always triggers Anesthesiology
    if urgency == "critical" and "Anesthesiology" not in seen:
        unique.append("Anesthesiology")

    # Fallback: nothing matched → General Medicine
    if not unique:
        unique = ["General Medicine"]

    reasoning_msg = (
        f"Router: activated {len(unique)} specialist(s) for patient {state['patient_id']}: "
        f"{unique} [urgency={urgency}]"
    )

    return {
        "active_specialists": unique,
        "reasoning": [reasoning_msg],
        "step_count": 1,
    }


# Inter-agent message helpers (used by specialist nodes)

def build_message(
    from_specialist: str,
    to_specialist: str,
    message_type: str,
    content: str,
) -> dict:
    """
    Build a typed inter-agent message.
    message_type: "finding" | "consultation_request" | "alert" | "response"
    """
    return {
        "from": from_specialist,
        "to": to_specialist,         # "ALL" for broadcast
        "type": message_type,
        "content": content,
    }


def get_messages_for(specialist: str, messages: list[dict]) -> list[dict]:
    """Filter inter_agent_messages addressed to a specific specialist or broadcast."""
    return [m for m in messages if m.get("to") in (specialist, "ALL")]
