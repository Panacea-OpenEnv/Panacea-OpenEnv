from typing import TypedDict, Annotated
import operator


class PanaceaState(TypedDict):
    """
    Shared state flowing through every node in the LangGraph episode graph.

    Append-only fields (sql_queries, query_results, reasoning) use operator.add
    so parallel or sequential nodes can contribute entries without clobbering each other.
    """

    # ── Claim intake ──────────────────────────────────────────────────────────
    claim_id: int
    patient_id: str
    department: str
    requested_resource: str
    claimed_amount: float
    proposed_claim: dict          # Raw dict as received from the sub-agent

    # ── SQL verification trail ────────────────────────────────────────────────
    sql_queries: Annotated[list, operator.add]    # Every query the agent issued
    query_results: Annotated[list, operator.add]  # Corresponding results/errors

    # ── Schema drift tracking ─────────────────────────────────────────────────
    schema_error_count: int       # How many ProgrammingErrors encountered this episode
    schema_adapted: bool          # True once the agent successfully recovered via information_schema
    known_tables: list            # Discovered table names after schema probe

    # ── Proactive check flags ─────────────────────────────────────────────────
    ghost_patient: bool           # True if patient_id not found in registry
    comorbidities_checked: bool   # True once proactive comorbidity scan ran

    # ── Analysis outputs ──────────────────────────────────────────────────────
    deception_detected: bool
    deception_type: str           # "none" | "ghost" | "inflation" | "masking" | "collusion"
    claimed_severity: float       # Severity implied by sub-agent's claim
    actual_severity: float        # Ground-truth severity from DB

    # ── Decision ──────────────────────────────────────────────────────────────
    verification_status: str      # "pending" | "verified_clean" | "deception_detected" | "cannot_verify"
    final_decision: str           # "APPROVED" | "REJECTED" | "PENDING"
    reasoning: Annotated[list, operator.add]  # Human-readable audit chain

    # ── Reward & telemetry ────────────────────────────────────────────────────
    reward: float
    step_count: int


def initial_state() -> PanaceaState:
    """Returns a blank episode state ready for the first LangGraph node."""
    return PanaceaState(
        claim_id=0,
        patient_id="",
        department="",
        requested_resource="",
        claimed_amount=0.0,
        proposed_claim={},
        sql_queries=[],
        query_results=[],
        schema_error_count=0,
        schema_adapted=False,
        known_tables=[],
        ghost_patient=False,
        comorbidities_checked=False,
        deception_detected=False,
        deception_type="none",
        claimed_severity=0.0,
        actual_severity=0.0,
        verification_status="pending",
        final_decision="PENDING",
        reasoning=[],
        reward=0.0,
        step_count=0,
    )
