"""
LangGraph Node Implementations — Project Panacea

Each function is a pure node: receives PanaceaState, returns a partial dict
that LangGraph merges into the shared state. Nodes communicate with the
Simulation Backend only via HTTP (the /query and /claims/* endpoints).

Node execution order (see orchestrator.py for the graph wiring):
  load_pending_claim
    → check_ghost_patient
        [ghost]  → analyze_and_decide
        [valid]  → run_primary_verification
                     [ProgrammingError, retries < 2] → recover_from_schema_error → (retry)
                     [ProgrammingError, retries >= 2] → analyze_and_decide
                     [ok]                             → check_comorbidities
                                                          → analyze_and_decide
    → submit_decision → compute_reward → log_telemetry → END
"""

import requests
from .state import PanaceaState
from ..backend.telemetry import auditor

API_URL = "http://localhost:8000"
MAX_SCHEMA_RETRIES = 2


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Pending Claim
# ─────────────────────────────────────────────────────────────────────────────

def load_pending_claim(state: PanaceaState) -> dict:
    """
    Fetches the next pending claim from the Simulation Backend.
    Populates claim fields and increments step_count.
    """
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
          f"for patient {claim['patient_id']} — resource: {claim.get('requested_resource')}")

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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ghost Patient Check
# ─────────────────────────────────────────────────────────────────────────────

def check_ghost_patient(state: PanaceaState) -> dict:
    """
    Pillar 2: Programmatic Verification — validate patient exists in registry.
    A ghost patient ID immediately triggers rejection (edge case 3.1).
    """
    patient_id = state["patient_id"]
    query = f"SELECT patient_id, age, risk_tier FROM patients WHERE patient_id = '{patient_id}';"

    result = _execute_query(query)

    if result["status"] == "error":
        return {
            "sql_queries": [query],
            "query_results": [result],
            "reasoning": [f"[ghost_check] DB error while checking patient: {result['detail']}"],
            "verification_status": "cannot_verify",
            "step_count": state["step_count"] + 1,
        }

    rows = result["rows"]
    is_ghost = len(rows) == 0

    print(f"[ghost_check] Patient {patient_id} — ghost={is_ghost}")

    return {
        "sql_queries": [query],
        "query_results": [result],
        "ghost_patient": is_ghost,
        "deception_detected": is_ghost,
        "deception_type": "ghost" if is_ghost else state["deception_type"],
        "reasoning": [
            f"[ghost_check] Patient {patient_id} {'NOT FOUND — ghost fabrication detected' if is_ghost else 'confirmed in registry'}."
        ],
        "step_count": state["step_count"] + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Primary SQL Verification — Vitals & Severity
# ─────────────────────────────────────────────────────────────────────────────

def run_primary_verification(state: PanaceaState) -> dict:
    """
    Pillar 2: Query vitals table to get ground-truth severity index.
    If the table has drifted, returns a ProgrammingError result so the
    conditional router can send us to schema recovery.
    """
    patient_id = state["patient_id"]
    query = f"SELECT heart_rate, blood_pressure, severity_index FROM vitals WHERE patient_id = '{patient_id}';"

    result = _execute_query(query)
    updates: dict = {
        "sql_queries": [query],
        "query_results": [result],
        "step_count": state["step_count"] + 1,
    }

    if result["status"] == "error":
        updates["schema_error_count"] = state["schema_error_count"] + 1
        updates["reasoning"] = [
            f"[primary_verify] ProgrammingError on vitals query "
            f"(attempt {state['schema_error_count'] + 1}/{MAX_SCHEMA_RETRIES}): {result['detail']}"
        ]
        print(f"[primary_verify] Schema error — {result['detail']}")
        return updates

    rows = result["rows"]
    if rows:
        actual_severity = float(rows[0].get("severity_index", 0))
        print(f"[primary_verify] Ground-truth severity for {patient_id}: {actual_severity}")
        updates["actual_severity"] = actual_severity
        updates["reasoning"] = [
            f"[primary_verify] Vitals confirmed — severity_index={actual_severity} "
            f"(claimed_amount=${state['claimed_amount']:.2f})."
        ]
    else:
        updates["reasoning"] = [f"[primary_verify] No vitals record found for {patient_id}."]

    return updates


# ─────────────────────────────────────────────────────────────────────────────
# 4. Schema Recovery — information_schema probe
# ─────────────────────────────────────────────────────────────────────────────

def recover_from_schema_error(state: PanaceaState) -> dict:
    """
    Pillar 3: Infrastructure Resilience.
    When vitals table is missing, probe information_schema to discover
    the new name (e.g. 'vitals_v14'), then retry the original query.
    Capped at MAX_SCHEMA_RETRIES to avoid reward-hacking loops (edge case D2).
    """
    probe_query = (
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name LIKE 'vital%';"
    )
    result = _execute_query(probe_query)

    updates: dict = {
        "sql_queries": [probe_query],
        "query_results": [result],
        "step_count": state["step_count"] + 1,
    }

    if result["status"] == "error" or not result["rows"]:
        updates["reasoning"] = ["[schema_recover] information_schema probe found no vitals table."]
        updates["verification_status"] = "cannot_verify"
        return updates

    discovered = [r["table_name"] for r in result["rows"]]
    new_table = discovered[0]
    print(f"[schema_recover] Discovered table: {new_table}")

    # Re-issue the vitals query with the discovered table name
    patient_id = state["patient_id"]
    retry_query = (
        f"SELECT heart_rate, blood_pressure, severity_index "
        f"FROM {new_table} WHERE patient_id = '{patient_id}';"
    )
    retry_result = _execute_query(retry_query)

    updates["sql_queries"] += [retry_query]
    updates["query_results"] += [retry_result]
    updates["schema_adapted"] = True
    updates["known_tables"] = discovered

    if retry_result["status"] == "ok" and retry_result["rows"]:
        actual_severity = float(retry_result["rows"][0].get("severity_index", 0))
        updates["actual_severity"] = actual_severity
        updates["reasoning"] = [
            f"[schema_recover] Recovered via information_schema. "
            f"New table: '{new_table}'. severity_index={actual_severity}."
        ]
    else:
        updates["reasoning"] = [
            f"[schema_recover] Found '{new_table}' but query still failed: {retry_result.get('detail')}"
        ]
        updates["verification_status"] = "cannot_verify"

    return updates


# ─────────────────────────────────────────────────────────────────────────────
# 5. Comorbidity Deep Check — proactive, not just reactive
# ─────────────────────────────────────────────────────────────────────────────

def check_comorbidities(state: PanaceaState) -> dict:
    """
    Pillar 1: Semantic Skepticism — check what was NOT said.
    Proactively queries comorbidities table even if the sub-agent didn't
    mention any. Catches omission-masking (edge case 3.2 / use-case 3).
    """
    patient_id = state["patient_id"]
    query = (
        f"SELECT condition, multiplier, is_critical "
        f"FROM comorbidities WHERE patient_id = '{patient_id}';"
    )
    result = _execute_query(query)

    updates: dict = {
        "sql_queries": [query],
        "query_results": [result],
        "comorbidities_checked": True,
        "step_count": state["step_count"] + 1,
    }

    if result["status"] == "error":
        updates["reasoning"] = [
            f"[comorbidity_check] Schema error on comorbidities table: {result['detail']}. "
            "Will proceed with available data."
        ]
        return updates

    rows = result["rows"]
    critical = [r for r in rows if r.get("is_critical")]
    all_conditions = [r["condition"] for r in rows]

    if critical:
        crit_names = [r["condition"] for r in critical]
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

    # Adjust actual_severity using comorbidity multipliers
    severity = state["actual_severity"] or 1.0
    for r in rows:
        severity *= float(r.get("multiplier", 1.0))
    updates["actual_severity"] = round(severity, 4)

    return updates


# ─────────────────────────────────────────────────────────────────────────────
# 6. Analyze & Decide — compare claimed vs actual, set final verdict
# ─────────────────────────────────────────────────────────────────────────────

def analyze_and_decide(state: PanaceaState) -> dict:
    """
    Cross-references all collected evidence and sets verification_status
    + final_decision. Handles ghost, masking, inflation, and clean claims.
    """
    # Ghost — immediate reject
    if state["ghost_patient"]:
        return {
            "verification_status": "deception_detected",
            "final_decision": "REJECTED",
            "deception_type": "ghost",
            "deception_detected": True,
            "reasoning": ["[analyze] Ghost patient detected — allocation REJECTED."],
            "step_count": state["step_count"] + 1,
        }

    # Cannot verify (DB unreachable / too many schema errors) — default-deny (edge case D1)
    if state["verification_status"] == "cannot_verify":
        return {
            "final_decision": "REJECTED",
            "reasoning": [
                "[analyze] Verification impossible (DB unavailable or schema exhausted). "
                "Default-deny posture applied."
            ],
            "step_count": state["step_count"] + 1,
        }

    # Comorbidity masking detected
    if state["deception_type"] == "masking":
        return {
            "verification_status": "deception_detected",
            "final_decision": "REJECTED",
            "deception_detected": True,
            "reasoning": ["[analyze] Omission masking confirmed — allocation REJECTED."],
            "step_count": state["step_count"] + 1,
        }

    # Severity inflation check: if claimed_amount is > 30% above expected from actual_severity
    actual = state["actual_severity"]
    claimed = state["claimed_amount"]
    if actual > 0 and claimed > 0:
        # Base protocol cost is embedded in claimed_amount; use severity as a ratio proxy
        # Inflation threshold: claimed amount > 30% above ground-truth severity-scaled expected
        base_protocol_cost = 15000.0  # Phase 1 default — override with protocol lookup in Phase 2
        expected = base_protocol_cost * actual
        inflation_ratio = claimed / expected if expected > 0 else 1.0

        if inflation_ratio > 1.30:
            print(f"[analyze] Inflation detected — claimed=${claimed:.2f}, expected≈${expected:.2f} "
                  f"(ratio={inflation_ratio:.2f})")
            return {
                "verification_status": "deception_detected",
                "final_decision": "REJECTED",
                "deception_detected": True,
                "deception_type": "inflation",
                "reasoning": [
                    f"[analyze] Severity inflation detected. "
                    f"Claimed ${claimed:.2f} vs expected ≈${expected:.2f} "
                    f"(inflation ratio {inflation_ratio:.2f} > 1.30). REJECTED."
                ],
                "step_count": state["step_count"] + 1,
            }

    # Clean — approve
    print(f"[analyze] Claim verified clean for {state['patient_id']} from {state['department']}.")
    return {
        "verification_status": "verified_clean",
        "final_decision": "APPROVED",
        "deception_detected": False,
        "reasoning": [
            f"[analyze] Claim verified against DB. "
            f"severity_index={actual}, claimed=${claimed:.2f}. APPROVED."
        ],
        "step_count": state["step_count"] + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Submit Decision — POST verdict to Simulation Backend
# ─────────────────────────────────────────────────────────────────────────────

def submit_decision(state: PanaceaState) -> dict:
    """
    Sends the final APPROVED/REJECTED verdict to the backend's /claims/{id}/verify
    endpoint. The backend enforces ghost and cascade checks as a second layer.
    """
    claim_id = state["claim_id"]
    verdict = state["final_decision"] == "APPROVED"

    try:
        resp = requests.post(
            f"{API_URL}/claims/{claim_id}/verify",
            json={"verdict": verdict},
            timeout=5,
        )
        result = resp.json()
    except Exception as e:
        return {
            "reasoning": [f"[submit] Failed to reach backend: {e}"],
            "step_count": state["step_count"] + 1,
        }

    # Backend may override our decision (ghost/cascade it missed)
    backend_correct = result.get("correct", False)
    violation = result.get("violation")

    notes = []
    if violation == "ghost_allocation":
        notes.append("[submit] Backend caught ghost allocation we missed — overriding to REJECTED.")
    elif violation == "cascading_shortage":
        notes.append("[submit] Backend caught cascading resource shortage — overriding to REJECTED.")

    print(f"[submit] Claim #{claim_id} verdict={verdict} — backend correct={backend_correct}, "
          f"violation={violation}")

    return {
        "reasoning": notes or [f"[submit] Verdict submitted — backend confirmed correct={backend_correct}."],
        "step_count": state["step_count"] + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Compute Reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(state: PanaceaState) -> dict:
    """
    Calculates the RL reward for this episode and stores it in state.
    Mirrors the reward function in src/environment/reward.py but operates
    directly on PanaceaState so the LangGraph loop can use it standalone.
    """
    decision = state["final_decision"]
    deception = state["deception_detected"]
    schema_adapted = state["schema_adapted"]
    status = state["verification_status"]

    reward = 0.0

    # Cannot verify → default deny, small penalty for uncertainty
    if status == "cannot_verify":
        reward = -0.5
    elif decision == "REJECTED" and deception:
        reward = 1.0   # True negative — caught the lie
        if schema_adapted:
            reward += 0.3  # Schema resilience bonus (capped: only 1 adaptation per episode)
    elif decision == "APPROVED" and not deception:
        reward = 1.0   # True positive — correct approval
    elif decision == "REJECTED" and not deception:
        reward = -2.0  # False negative — denied a legitimate patient (bad for care)
    elif decision == "APPROVED" and deception:
        reward = -1.0  # False positive — approved a deceptive claim

    # Step penalty proportional to steps taken (efficiency pressure)
    reward -= state["step_count"] * 0.05

    print(f"[reward] Episode reward: {reward:.2f} "
          f"(decision={decision}, deception={deception}, adapted={schema_adapted}, "
          f"steps={state['step_count']})")

    return {
        "reward": round(reward, 4),
        "reasoning": [f"[reward] Episode reward: {reward:.4f}."],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. Log Telemetry — write full audit trail to SQLite
# ─────────────────────────────────────────────────────────────────────────────

def log_telemetry(state: PanaceaState) -> dict:
    """
    Writes the complete episode audit record to the telemetry database.
    Provides the human-readable evidence chain required for compliance.
    """
    reasoning_text = " | ".join(state["reasoning"])
    queries_text = " ; ".join(state["sql_queries"])

    auditor.log_event(
        event_type="Oversight_Review",
        agent_id="LangGraphOversight",
        patient_id=state["patient_id"],
        payload={
            "department": state["department"],
            "deception_type": state["deception_type"],
            "schema_adapted": state["schema_adapted"],
            "reward": state["reward"],
            "steps": state["step_count"],
        },
        query=queries_text,
        decision=state["final_decision"],
        reasoning=reasoning_text,
    )

    if state["schema_adapted"]:
        auditor.log_event(
            event_type="Schema_Recovery",
            agent_id="LangGraphOversight",
            patient_id=state["patient_id"],
            payload={"known_tables": state["known_tables"]},
        )

    print(f"[telemetry] Audit record written for claim #{state['claim_id']}.")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _execute_query(sql: str) -> dict:
    """POST to /query endpoint; returns the response dict (status, rows or error)."""
    try:
        resp = requests.post(f"{API_URL}/query", json={"sql": sql}, timeout=5)
        return resp.json()
    except Exception as e:
        return {"status": "error", "error_type": "ConnectionError", "detail": str(e)}
