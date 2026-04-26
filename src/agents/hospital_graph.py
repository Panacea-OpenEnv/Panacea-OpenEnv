"""
Hospital Multi-Agent LangGraph Graph

Architecture:
  patient_intake → triage_router
    → [Send fan-out] → run_specialist × N  (parallel, one per active specialty)
    → consultation_check
    → [Send fan-out again if consultations pending] → run_specialist × M
    → synthesize_treatment_plan
    → oversight_verify
    → final_decision → END

Key LangGraph features used:
  - StateGraph with HospitalState (TypedDict + operator.add reducers)
  - Send API for parallel fan-out to specialist nodes
  - Conditional edges for consultation loop (max 2 rounds)
  - Append-only list fields so parallel nodes never clobber each other
"""

import os
import requests
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from .hospital_state import HospitalState, initial_hospital_state
from .router_agent import patient_intake, triage_router, build_message, get_messages_for
from .specialists.registry import SPECIALISTS, get_specialist

BACKEND_URL = "http://localhost:8000"
MAX_CONSULTATION_ROUNDS = 2


# Specialist node — runs for each specialist dispatched via Send

def _query_db(sql: str) -> list[dict]:
    """Helper: run a read-only SQL query against the backend."""
    try:
        resp = requests.post(f"{BACKEND_URL}/query", json={"sql": sql}, timeout=5)
        if resp.ok:
            return resp.json().get("rows", [])
    except Exception:
        pass
    return []


def _assess_patient(state: HospitalState, spec_name: str, spec_data: dict) -> dict:
    """
    Core domain assessment logic for a specialist.

    Queries the DB for patient-relevant data, checks conditions that match
    this specialty, determines resource needs, detects if consultations are
    required, and builds an inter-agent message for any critical finding.
    """
    patient_id = state["patient_id"]
    conditions = state.get("conditions") or []
    symptoms = state.get("symptoms") or []
    urgency = state.get("urgency", "medium")

    #  Identify which of this specialist's conditions the patient has 
    specialty_conditions = spec_data["conditions"]
    matched_conditions = [
        c for c in conditions
        if any(c.lower().replace(" ", "_") in sc for sc in specialty_conditions)
           or any(sc in c.lower().replace(" ", "_") for sc in specialty_conditions)
    ]

    #  Query relevant vitals from DB 
    vitals_rows = _query_db(
        f"SELECT * FROM vitals WHERE patient_id = '{patient_id}' LIMIT 1"
    )
    vitals = vitals_rows[0] if vitals_rows else {}

    #  Determine severity score (0.0–1.0) 
    base_severity = len(matched_conditions) * 0.2 + (0.3 if urgency == "critical" else 0.1)
    severity = min(base_severity, 1.0)
    severity_label = (
        "critical" if severity >= 0.7
        else "high" if severity >= 0.5
        else "moderate" if severity >= 0.3
        else "mild"
    )

    #  Select resources to request (scale with severity) 
    available_resources = spec_data["resources"]
    n_resources = max(1, int(len(available_resources) * severity))
    requested_resources = available_resources[:n_resources]

    #  Decide which consultations to request 
    consultation_requests: list[str] = []
    completed = set(state.get("completed_specialists") or [])
    already_active = set(state.get("active_specialists") or [])

    if severity >= 0.5:
        # Request top consultation partners not already running
        for partner in spec_data["consultation_partners"][:2]:
            if partner not in completed and partner not in already_active:
                consultation_requests.append(partner)

    #  Read messages from other agents directed to this specialist 
    incoming = get_messages_for(spec_name, state.get("inter_agent_messages") or [])
    incoming_summary = (
        f"; received {len(incoming)} inter-agent message(s)" if incoming else ""
    )

    #  Build outgoing inter-agent messages 
    outgoing_messages: list[dict] = []
    if matched_conditions and severity >= 0.5:
        for partner in spec_data["consultation_partners"][:1]:
            outgoing_messages.append(build_message(
                from_specialist=spec_name,
                to_specialist=partner,
                message_type="finding",
                content=(
                    f"{spec_data['role']} reports {severity_label} severity "
                    f"for conditions {matched_conditions} in patient {patient_id}. "
                    f"Requesting review."
                ),
            ))

    #  Build the report dict 
    report = {
        "specialist": spec_name,
        "role": spec_data["role"],
        "patient_id": patient_id,
        "matched_conditions": matched_conditions,
        "severity": severity,
        "severity_label": severity_label,
        "vitals_snapshot": vitals,
        "requested_resources": requested_resources,
        "consultation_requests": consultation_requests,
        "assessment": (
            f"{spec_data['role']} assessment: patient presents with "
            f"{matched_conditions or symptoms[:2]} at {severity_label} severity. "
            f"Requesting {requested_resources[:2]}.{incoming_summary}"
        ),
        "protocol": spec_data["protocols"][0] if spec_data["protocols"] else "standard_care",
    }

    return {
        "report": report,
        "resource_requests": [{
            "specialist": spec_name,
            "resources": requested_resources,
            "severity": severity,
            "urgency": urgency,
        }],
        "consultation_requests": consultation_requests,
        "outgoing_messages": outgoing_messages,
    }


def run_specialist(state: HospitalState) -> dict:
    """
    LangGraph node — runs for each specialist dispatched by Send.
    Reads state["current_specialist"] to know which doctor this is.
    """
    spec_name = state.get("current_specialist", "")
    if not spec_name or spec_name not in SPECIALISTS:
        return {
            "reasoning": [f"run_specialist: unknown specialist '{spec_name}', skipping"],
        }

    spec_data = get_specialist(spec_name)
    result = _assess_patient(state, spec_name, spec_data)

    report = result["report"]
    consultations = result["consultation_requests"]

    return {
        "specialist_reports": [report],
        "completed_specialists": [spec_name],
        "pending_consultations": consultations,
        "resource_requests": result["resource_requests"],
        "inter_agent_messages": result["outgoing_messages"],
        "reasoning": [
            f"{spec_name} ({spec_data['role']}): {report['severity_label']} severity "
            f"— requesting {report['requested_resources'][:2]}"
            + (f", consulting {consultations}" if consultations else "")
        ],
        "step_count": 1,
    }


# Treatment plan synthesizer

def synthesize_treatment_plan(state: HospitalState) -> dict:
    """
    Merge all specialist reports into a unified treatment plan.
    Identifies primary lead specialist (highest severity), consolidates
    resources, and flags any conflicts.
    """
    reports = state.get("specialist_reports") or []
    if not reports:
        return {
            "treatment_plan": {"status": "no_specialists_ran", "resources": []},
            "reasoning": ["Synthesizer: no specialist reports to merge"],
        }

    # Find lead specialist (highest severity)
    lead = max(reports, key=lambda r: r.get("severity", 0))

    # Consolidate all requested resources (deduplicated)
    all_resources: list[str] = []
    seen_res: set[str] = set()
    for r in reports:
        for res in r.get("requested_resources", []):
            if res not in seen_res:
                seen_res.add(res)
                all_resources.append(res)

    # Collect all conditions addressed
    all_conditions: list[str] = []
    for r in reports:
        all_conditions.extend(r.get("matched_conditions", []))

    treatment_plan = {
        "lead_specialist": lead["specialist"],
        "lead_role": lead["role"],
        "lead_severity": lead["severity_label"],
        "specialists_involved": [r["specialist"] for r in reports],
        "conditions_addressed": list(dict.fromkeys(all_conditions)),
        "consolidated_resources": all_resources,
        "primary_protocol": lead.get("protocol", "standard_care"),
        "specialist_count": len(reports),
    }

    reasoning = (
        f"Synthesizer: {len(reports)} specialist(s) assessed patient "
        f"{state['patient_id']}. Lead: {lead['role']} "
        f"({lead['severity_label']} severity). "
        f"Total resources requested: {len(all_resources)}."
    )

    return {
        "treatment_plan": treatment_plan,
        "reasoning": [reasoning],
        "step_count": 1,
    }


# Oversight verifier

def oversight_verify(state: HospitalState) -> dict:
    """
    Cross-check all specialist resource requests against the DB.
    Calls verify_claim from oversight_core.
    """
    from .oversight_core import verify_claim
    patient_id = state["patient_id"]
    reports = state.get("specialist_reports") or []
    resource_requests = state.get("resource_requests") or []

    # Ghost patient check requires patient_from_db
    patient_rows = _query_db(f"SELECT patient_id FROM patients WHERE patient_id = '{patient_id}'")
    patient_from_db = patient_rows[0] if patient_rows else None

    verify_result = verify_claim(
        patient_id=patient_id,
        reports=reports,
        resource_requests=resource_requests,
        patient_from_db=patient_from_db,
    )

    return {
        "oversight_decision": verify_result["decision"],
        "fraud_flags": verify_result["fraud_flags"],
        "reasoning": [verify_result["reasoning"]],
        "step_count": 1,
    }


# Final decision

def final_decision(state: HospitalState) -> dict:
    """Compute reward and write final summary to reasoning trail."""
    decision = state.get("oversight_decision", "PENDING")
    fraud_flags = state.get("fraud_flags") or []
    reports = state.get("specialist_reports") or []

    # Reward: +1 per clean specialist verified, -2 per fraud flag
    reward = len(reports) * 0.5 - len(fraud_flags) * 2.0
    reward = max(-10.0, min(reward, 10.0))  # clamp

    summary = (
        f"FINAL: decision={decision}, specialists={len(reports)}, "
        f"fraud_flags={len(fraud_flags)}, reward={reward:.2f}"
    )

    return {
        "reward": reward,
        "reasoning": [summary],
        "step_count": 1,
    }


# Conditional edge functions

def dispatch_to_specialists(state: HospitalState) -> list[Send]:
    """
    Fan-out: send the patient state to each active specialist in parallel.
    Called as a conditional edge after triage_router.
    """
    active = state.get("active_specialists") or []
    if not active:
        # No specialists — jump straight to synthesis
        return [Send("synthesize_treatment_plan", state)]
    return [
        Send("run_specialist", {**state, "current_specialist": spec})
        for spec in active
    ]


def consultation_gateway(_state: HospitalState) -> dict:
    """
    Fan-in node: sits after all parallel run_specialist instances.
    Forces LangGraph to merge all specialist outputs before deciding
    whether to dispatch consultation specialists.
    Returns nothing — just a merge point.
    """
    return {}


def dispatch_consultations(state: HospitalState) -> list[Send] | str:
    """
    Called as conditional edge from consultation_gateway.
    Uses a separate node name (run_consultation) so the graph never cycles
    back through the initial fan-out path.
    """
    completed = set(state.get("completed_specialists") or [])
    pending = [
        s for s in (state.get("pending_consultations") or [])
        if s not in completed and s in SPECIALISTS
    ]
    if pending:
        return [
            Send("run_consultation", {**state, "current_specialist": spec})
            for spec in pending
        ]
    return "synthesize_treatment_plan"


# Graph assembly

def build_hospital_graph() -> StateGraph:
    graph = StateGraph(HospitalState)

    # Register nodes
    graph.add_node("patient_intake",            patient_intake)
    graph.add_node("triage_router",             triage_router)
    graph.add_node("run_specialist",            run_specialist)
    # run_consultation is the SAME function but a distinct node name —
    # this prevents cycles: initial specialists → gateway → consultations → synthesis
    graph.add_node("run_consultation",          run_specialist)
    graph.add_node("consultation_gateway",      consultation_gateway)
    graph.add_node("synthesize_treatment_plan", synthesize_treatment_plan)
    graph.add_node("oversight_verify",          oversight_verify)
    graph.add_node("final_decision",            final_decision)

    # Entry
    graph.set_entry_point("patient_intake")

    # intake → router
    graph.add_edge("patient_intake", "triage_router")

    # router → fan-out to initial specialists (via Send)
    graph.add_conditional_edges("triage_router", dispatch_to_specialists)

    # initial specialists all merge into consultation_gateway (fan-in)
    graph.add_edge("run_specialist", "consultation_gateway")

    # gateway decides: more consultations needed? or proceed to synthesis
    graph.add_conditional_edges("consultation_gateway", dispatch_consultations)

    # consultation specialists go straight to synthesis (no further loops)
    graph.add_edge("run_consultation", "synthesize_treatment_plan")

    # synthesis → oversight → final → END
    graph.add_edge("synthesize_treatment_plan", "oversight_verify")
    graph.add_edge("oversight_verify",          "final_decision")
    graph.add_edge("final_decision",            END)

    return graph


# Compiled graph — importable by training harness and demo
hospital_app = build_hospital_graph().compile()


# Episode runner

def run_hospital_episode(
    patient_id: str = "P1001",
    symptoms: list[str] | None = None,
    conditions: list[str] | None = None,
    urgency: str = "medium",
) -> HospitalState:
    """
    Run one full hospital multi-agent episode.

    Example:
        result = run_hospital_episode(
            patient_id="P1003",
            symptoms=["chest pain", "shortness of breath"],
            conditions=["myocardial_infarction", "diabetes_mellitus"],
            urgency="critical",
        )
    """
    state = initial_hospital_state(
        patient_id=patient_id,
        symptoms=symptoms or [],
        conditions=conditions or [],
        urgency=urgency,
    )

    print("\n" + "=" * 70)
    print("  PANACEA -- Hospital Multi-Agent Episode Starting")
    print(f"  Patient: {patient_id} | Urgency: {urgency}")
    print(f"  Symptoms: {symptoms}")
    print(f"  Conditions: {conditions}")
    print("=" * 70)

    result = hospital_app.invoke(state)

    plan = result.get("treatment_plan", {})
    print("\n" + "-" * 70)
    print(f"  Decision       : {result['oversight_decision']}")
    print(f"  Lead Specialist: {plan.get('lead_role', 'N/A')} ({plan.get('lead_severity', 'N/A')})")
    print(f"  Specialists Run: {result.get('completed_specialists', [])}")
    print(f"  Fraud Flags    : {result.get('fraud_flags', [])}")
    print(f"  Reward         : {result['reward']:.2f}")
    print(f"  Steps          : {result['step_count']}")
    print("-" * 70)
    print("\nReasoning chain:")
    for step in result.get("reasoning", []):
        print(f"  {step}")

    return result


if __name__ == "__main__":
    run_hospital_episode(
        patient_id="P1001",
        symptoms=["chest pain", "shortness of breath"],
        conditions=["myocardial_infarction", "diabetes_mellitus"],
        urgency="critical",
    )
