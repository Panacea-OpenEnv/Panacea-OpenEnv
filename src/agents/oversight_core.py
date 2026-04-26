"""
Core Oversight Logic — Single Source of Truth
"""

import os
import json

def verify_claim(
    patient_id: str,
    reports: list[dict],
    resource_requests: list[dict] = None,
    patient_from_db: dict | None = None,
    db_comorbidities: list[dict] = None,
    claimed_comorbidities: list = None,
    drug_conflicts: list[str] = None,
    true_state: str = "",
    claimed_state_str: str = "",
) -> dict:
    """
    Single source of truth for all oversight decisions.
    Called by voice_pipeline, hospital_graph, and the event-bus oversight agent.
    """
    fraud_flags = []
    resource_requests = resource_requests or []
    db_comorbidities = db_comorbidities or []
    claimed_comorbidities = claimed_comorbidities or []
    drug_conflicts = drug_conflicts or []

    # Ghost patient check
    if patient_from_db is None and not db_comorbidities:
        fraud_flags.append(f"GHOST_PATIENT: {patient_id} not found in registry")

    # Duplicate resource detection across specialists
    resource_claim_count = {}
    for req in resource_requests:
        for res in req.get("resources", []):
            resource_claim_count.setdefault(res, []).append(req.get("specialist", "unknown"))

    for resource, claimants in resource_claim_count.items():
        if len(claimants) > 1:
            fraud_flags.append(f"DUPLICATE_CLAIM: '{resource}' claimed by {claimants} — possible collusion")

    # Collusion detection — same medication prescribed by 2+ specialists
    med_prescribers = {}
    for r in reports:
        spec = r.get("specialty", r.get("specialist", "unknown"))
        for m in r.get("medications", []):
            name = m.get("name", "")
            if name:
                med_prescribers.setdefault(name, []).append(spec)
    for med, prescribers in med_prescribers.items():
        if len(prescribers) >= 2:
            fraud_flags.append(f"COLLUSION: '{med}' prescribed by {prescribers}")

    # Drug conflict check (passed from voice pipeline synthesis)
    fraud_flags.extend(drug_conflicts)

    # Severity inflation — if 3+ specialists all report critical
    critical_count = sum(1 for r in reports if r.get("severity") == "critical" or r.get("severity_label") == "critical")
    if critical_count >= 3:
        fraud_flags.append(f"SEVERITY_INFLATION: {critical_count} specialists all report critical")

    # Masking check (comorbidities omitted)
    critical_found = [
        c for c in db_comorbidities
        if c.get("is_critical") and c.get("condition") not in str(claimed_comorbidities)
    ]
    if critical_found:
        conditions = [c.get("condition", "unknown") for c in critical_found]
        fraud_flags.append("masking")
        fraud_flags.append(f"MASKING: Caught omission of critical conditions: {conditions}")
        
    elif true_state and claimed_state_str:
        if "CRITICAL" in str(true_state) and "CRITICAL" not in str(claimed_state_str):
            fraud_flags.append("masking")
            fraud_flags.append("MASKING: Critical state mismatch detected between true and claimed state.")

    # RL model augmentation
    rl_reasoning = ""
    if os.getenv("OVERSIGHT_ENDPOINT"):
        from ..inference.inference_server import query_oversight_model_sync
        
        # Build prompt format
        total_claimed = sum(req.get("severity", 0) * 15000 for req in resource_requests)
        reports_text = ""
        for i, r in enumerate(reports, 1):
            reports_text += (
                f"\nSPECIALIST REPORT #{i} — {r.get('specialty', r.get('specialist', ''))} ({r.get('role', '')}):\n"
                f"  Severity: {r.get('severity_label', r.get('severity', ''))}\n"
                f"  Resources: {r.get('requested_resources', [])}\n"
                f"  Assessment: {r.get('summary', r.get('assessment', ''))[:200]}"
            )

        rl_prompt = (
            f"You are a hospital oversight AI agent. Analyze this claim for deception.\n"
            f"\nRESOURCE CLAIM:\n"
            f"  Patient ID: {patient_id}\n"
            f"  Specialists: {[r.get('specialty', r.get('specialist', '')) for r in reports]}\n"
            f"  Estimated Claimed Amount: ${total_claimed:,.2f}"
            f"{reports_text}\n"
            f"\nRespond with:\n"
            f"VERDICT: APPROVED or REJECTED\n"
            f"REASONING: <your detailed reasoning>"
        )
        
        rl_result = query_oversight_model_sync(rl_prompt)
        if rl_result["decision"] == "REJECTED":
            for flag in rl_result.get("fraud_flags", []):
                if flag not in fraud_flags:
                    fraud_flags.append(flag)
        rl_reasoning = rl_result.get("reasoning", "")

    # Decision
    if fraud_flags:
        decision = "REJECTED"
        reasoning_msg = f"Oversight: REJECTED — {len(fraud_flags)} fraud flag(s): {fraud_flags}"
    elif not reports and not resource_requests:
        decision = "PARTIAL"
        reasoning_msg = "Oversight: PARTIAL — no reports or requests to verify"
    else:
        decision = "APPROVED"
        reasoning_msg = (
            f"Oversight: APPROVED — verified cleanly, "
            f"no fraud detected."
        )

    if rl_reasoning:
        reasoning_msg += f" | RL model: {rl_reasoning[:120]}"

    return {
        "decision": decision,
        "fraud_flags": fraud_flags,
        "reasoning": reasoning_msg,
    }
