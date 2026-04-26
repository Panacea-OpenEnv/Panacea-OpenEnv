"""
LangGraph Node Implementations — Project Panacea (MongoDB V2)
"""

import requests
from .state import PanaceaState
from ..backend.telemetry import auditor
from ..database.mongo_client import get_sync_collection, PATIENTS, VITALS, COMORBIDITIES

API_URL = "http://localhost:8000"

# Load Pending Claim

def load_pending_claim(state: PanaceaState) -> dict:
    try:
        resp = requests.get(f"{API_URL}/claims/pending", timeout=5)
        claims = resp.json()
    except Exception as e:
        return {
            "reasoning": [f"[load_claim] Backend unreachable: {e}"],
            "verification_status": "cannot_verify",
            "final_decision": "REJECTED",
            "step_count": state["step_count"] + 1,
        }

    if not claims:
        return {
            "reasoning": ["[load_claim] No pending claims in queue."],
            "verification_status": "cannot_verify",
            "final_decision": "REJECTED",
            "step_count": state["step_count"] + 1,
        }

    claim = claims[0]
    print(f"[load_claim] Claim #{claim['id']} from {claim['department']} "
          f"for patient {claim['patient_id']}")

    return {
        "claim_id": claim["id"],
        "patient_id": claim["patient_id"],
        "department": claim["department"],
        "requested_resource": claim.get("requested_resource", ""),
        "claimed_amount": float(claim.get("claimed_amount", 0)),
        "proposed_claim": claim,
        "reasoning": [f"[load_claim] Received claim #{claim['id']} from {claim['department']}."],
        "step_count": state["step_count"] + 1,
    }


# Ghost Patient Check

def check_ghost_patient(state: PanaceaState) -> dict:
    patient_id = state["patient_id"]
    try:
        patient = get_sync_collection(PATIENTS).find_one({"patient_id": patient_id})
    except Exception as e:
        return {
            "reasoning": [f"[ghost_check] DB error while checking patient: {e}"],
            "verification_status": "cannot_verify",
            "step_count": state["step_count"] + 1,
        }

    is_ghost = patient is None
    print(f"[ghost_check] Patient {patient_id} — ghost={is_ghost}")

    return {
        "ghost_patient": is_ghost,
        "deception_detected": is_ghost,
        "deception_type": "ghost" if is_ghost else state["deception_type"],
        "reasoning": [
            f"[ghost_check] Patient {patient_id} {'NOT FOUND — ghost fabrication detected' if is_ghost else 'confirmed in registry'}."
        ],
        "step_count": state["step_count"] + 1,
    }


# Primary Verification — Vitals & Severity

def run_primary_verification(state: PanaceaState) -> dict:
    patient_id = state["patient_id"]
    
    try:
        vitals = get_sync_collection(VITALS).find_one(
            {"patient_id": patient_id},
            sort=[("recorded_at", -1)]
        )
    except Exception as e:
        return {
            "reasoning": [f"[primary_verify] DB Error: {e}"],
            "verification_status": "cannot_verify",
            "step_count": state["step_count"] + 1,
        }

    updates: dict = {"step_count": state["step_count"] + 1}

    if vitals:
        actual_severity = float(vitals.get("severity_index", 0))
        print(f"[primary_verify] Ground-truth severity for {patient_id}: {actual_severity}")
        updates["actual_severity"] = actual_severity
        updates["reasoning"] = [
            f"[primary_verify] Vitals confirmed — severity_index={actual_severity}."
        ]
    else:
        updates["reasoning"] = [f"[primary_verify] No vitals record found for {patient_id}."]

    return updates


# Comorbidity Deep Check

def check_comorbidities(state: PanaceaState) -> dict:
    patient_id = state["patient_id"]
    
    try:
        rows = list(get_sync_collection(COMORBIDITIES).find({"patient_id": patient_id}))
    except Exception as e:
        return {
            "reasoning": [f"[comorbidity_check] Error fetching comorbidities: {e}"],
            "step_count": state["step_count"] + 1,
        }

    updates: dict = {"comorbidities_checked": True, "step_count": state["step_count"] + 1}

    critical = [r for r in rows if r.get("is_critical")]
    all_conditions = [r.get("condition") for r in rows]

    if critical:
        crit_names = [r.get("condition") for r in critical]
        updates["deception_detected"] = True
        updates["deception_type"] = "masking"
        updates["reasoning"] = [
            f"[comorbidity_check] CRITICAL comorbidities found: {crit_names}. "
            "These were omitted from the sub-agent's claim — masking deception confirmed."
        ]
        print(f"[comorbidity_check] Masking detected! Critical conditions: {crit_names}")
    else:
        updates["reasoning"] = [
            f"[comorbidity_check] No critical comorbidities for {patient_id}. "
            f"All conditions: {all_conditions or 'none'}."
        ]

    severity = state["actual_severity"] or 1.0
    for r in rows:
        severity *= float(r.get("multiplier", 1.0))
    updates["actual_severity"] = round(severity, 4)

    return updates


# Analyze & Decide

def analyze_and_decide(state: PanaceaState) -> dict:
    if state["ghost_patient"]:
        return {
            "verification_status": "deception_detected",
            "final_decision": "REJECTED",
            "deception_type": "ghost",
            "deception_detected": True,
            "reasoning": ["[analyze] Ghost patient detected — allocation REJECTED."],
            "step_count": state["step_count"] + 1,
        }

    if state["verification_status"] == "cannot_verify":
        return {
            "final_decision": "REJECTED",
            "reasoning": ["[analyze] Verification impossible. Default-deny posture applied."],
            "step_count": state["step_count"] + 1,
        }

    if state["deception_type"] == "masking":
        return {
            "verification_status": "deception_detected",
            "final_decision": "REJECTED",
            "deception_detected": True,
            "reasoning": ["[analyze] Omission masking confirmed — allocation REJECTED."],
            "step_count": state["step_count"] + 1,
        }

    actual = state["actual_severity"]
    claimed = state["claimed_amount"]
    if actual > 0 and claimed > 0:
        base_protocol_cost = 15000.0 
        expected = base_protocol_cost * actual
        inflation_ratio = claimed / expected if expected > 0 else 1.0

        if inflation_ratio > 1.30:
            print(f"[analyze] Inflation detected — ratio={inflation_ratio:.2f}")
            return {
                "verification_status": "deception_detected",
                "final_decision": "REJECTED",
                "deception_detected": True,
                "deception_type": "inflation",
                "reasoning": [
                    f"[analyze] Severity inflation detected. Claimed ${claimed:.2f} vs expected ≈${expected:.2f} REJECTED."
                ],
                "step_count": state["step_count"] + 1,
            }

    print(f"[analyze] Claim verified clean for {state['patient_id']}.")
    return {
        "verification_status": "verified_clean",
        "final_decision": "APPROVED",
        "deception_detected": False,
        "reasoning": ["[analyze] Claim verified against DB. APPROVED."],
        "step_count": state["step_count"] + 1,
    }


# Submit Decision

def submit_decision(state: PanaceaState) -> dict:
    claim_id = state["claim_id"]
    verdict = state["final_decision"] == "APPROVED"

    try:
        resp = requests.post(f"{API_URL}/claims/{claim_id}/verify", json={"verdict": verdict}, timeout=5)
        result = resp.json()
    except Exception as e:
        return {"reasoning": [f"[submit] Failed to reach backend: {e}"], "step_count": state["step_count"] + 1}

    backend_correct = result.get("correct", False)
    violation = result.get("violation")
    return {
        "reasoning": [f"[submit] Verdict submitted — backend confirmed correct={backend_correct} (violation={violation})."],
        "step_count": state["step_count"] + 1,
    }


# Compute Reward

def compute_reward(state: PanaceaState) -> dict:
    decision = state["final_decision"]
    deception = state["deception_detected"]
    status = state["verification_status"]

    reward = 0.0

    if status == "cannot_verify":
        reward = -0.5
    elif decision == "REJECTED" and deception:
        reward = 1.0
    elif decision == "APPROVED" and not deception:
        reward = 1.0
    elif decision == "REJECTED" and not deception:
        reward = -2.0
    elif decision == "APPROVED" and deception:
        reward = -1.0

    reward -= state["step_count"] * 0.05
    return {"reward": round(reward, 4), "reasoning": [f"[reward] Episode reward: {reward:.4f}."]}


# Log Telemetry

def log_telemetry(state: PanaceaState) -> dict:
    reasoning_text = " | ".join(state["reasoning"])
    auditor.log_event(
        event_type="Oversight_Review",
        agent_id="LangGraphOversight",
        patient_id=state["patient_id"],
        payload={
            "department": state["department"],
            "deception_type": state["deception_type"],
            "reward": state["reward"],
            "steps": state["step_count"],
        },
        query="MongoDB Read",
        decision=state["final_decision"],
        reasoning=reasoning_text,
    )
    return {}
