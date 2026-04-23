"""
Project Panacea — LangGraph Orchestrator

Assembles the stateful episode graph that runs one full verification cycle:
  load claim → ghost check → SQL verify (with schema-drift recovery loop)
  → comorbidity deep-check → analyze → submit → reward → telemetry → END

Conditional routing implements all three oversight pillars:
  Pillar 1 – Semantic Skepticism   : always runs comorbidity check, even if not mentioned
  Pillar 2 – Programmatic Verify   : real SQL via /query endpoint, not assertions
  Pillar 3 – Infrastructure Resil. : schema error loop (max 2 retries, capped reward)

Run a single episode:
  python -m src.agents.orchestrator
"""

from langgraph.graph import StateGraph, END

from .state import PanaceaState, initial_state
from .nodes import (
    load_pending_claim,
    check_ghost_patient,
    run_primary_verification,
    recover_from_schema_error,
    check_comorbidities,
    analyze_and_decide,
    submit_decision,
    compute_reward,
    log_telemetry,
)

MAX_SCHEMA_RETRIES = 2


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# ─────────────────────────────────────────────────────────────────────────────

def route_after_ghost_check(state: PanaceaState) -> str:
    """
    If patient is a ghost → skip SQL and go straight to decision (will REJECT).
    If we already cannot verify → go straight to decision.
    Otherwise → start primary SQL verification.
    """
    if state["ghost_patient"] or state["verification_status"] == "cannot_verify":
        return "analyze_and_decide"
    return "run_primary_verification"


def route_after_primary_verify(state: PanaceaState) -> str:
    """
    If the latest query result was a ProgrammingError AND we haven't exceeded
    MAX_SCHEMA_RETRIES, go to schema recovery (Pillar 3).
    Otherwise proceed to the comorbidity deep-check.
    """
    results = state["query_results"]
    last_result = results[-1] if results else {}
    is_programming_error = (
        last_result.get("status") == "error"
        and "does not exist" in last_result.get("detail", "")
    )

    if is_programming_error and state["schema_error_count"] < MAX_SCHEMA_RETRIES:
        return "recover_from_schema_error"

    # Too many retries or non-schema error → cannot verify
    if last_result.get("status") == "error":
        return "analyze_and_decide"

    return "check_comorbidities"


def route_after_schema_recovery(state: PanaceaState) -> str:
    """
    If recovery succeeded (actual_severity populated) → continue to comorbidities.
    If recovery failed → go to decision with cannot_verify status.
    """
    if state["verification_status"] == "cannot_verify":
        return "analyze_and_decide"
    if state["actual_severity"] > 0:
        return "check_comorbidities"
    return "analyze_and_decide"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PanaceaState)

    # Register all nodes
    graph.add_node("load_pending_claim",       load_pending_claim)
    graph.add_node("check_ghost_patient",      check_ghost_patient)
    graph.add_node("run_primary_verification", run_primary_verification)
    graph.add_node("recover_from_schema_error",recover_from_schema_error)
    graph.add_node("check_comorbidities",      check_comorbidities)
    graph.add_node("analyze_and_decide",       analyze_and_decide)
    graph.add_node("submit_decision",          submit_decision)
    graph.add_node("compute_reward",           compute_reward)
    graph.add_node("log_telemetry",            log_telemetry)

    # Entry point
    graph.set_entry_point("load_pending_claim")

    # Fixed edges
    graph.add_edge("load_pending_claim", "check_ghost_patient")

    # Conditional: ghost check result
    graph.add_conditional_edges(
        "check_ghost_patient",
        route_after_ghost_check,
        {
            "analyze_and_decide":       "analyze_and_decide",
            "run_primary_verification": "run_primary_verification",
        },
    )

    # Conditional: primary SQL result (ok | schema_error | other_error)
    graph.add_conditional_edges(
        "run_primary_verification",
        route_after_primary_verify,
        {
            "recover_from_schema_error": "recover_from_schema_error",
            "check_comorbidities":       "check_comorbidities",
            "analyze_and_decide":        "analyze_and_decide",
        },
    )

    # Conditional: schema recovery result
    graph.add_conditional_edges(
        "recover_from_schema_error",
        route_after_schema_recovery,
        {
            "check_comorbidities": "check_comorbidities",
            "analyze_and_decide":  "analyze_and_decide",
        },
    )

    # Fixed tail: comorbidities → decide → submit → reward → telemetry → END
    graph.add_edge("check_comorbidities",      "analyze_and_decide")
    graph.add_edge("analyze_and_decide",       "submit_decision")
    graph.add_edge("submit_decision",          "compute_reward")
    graph.add_edge("compute_reward",           "log_telemetry")
    graph.add_edge("log_telemetry",            END)

    return graph


# Compiled application — importable by training harness and demo script
app = build_graph().compile()


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner helper (used by training loop and demo)
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(override_state: dict | None = None) -> PanaceaState:
    """
    Executes one full verification episode and returns the final state.

    Args:
        override_state: Optional partial dict to inject into initial_state
                        (useful for testing specific scenarios).
    Returns:
        The completed PanaceaState after all nodes have run.
    """
    state = initial_state()
    if override_state:
        state.update(override_state)

    print("\n" + "═" * 60)
    print("  PROJECT PANACEA — LangGraph Episode Starting")
    print("═" * 60)

    result = app.invoke(state)

    print("\n" + "─" * 60)
    print(f"  Decision  : {result['final_decision']}")
    print(f"  Deception : {result['deception_type']} (detected={result['deception_detected']})")
    print(f"  Reward    : {result['reward']}")
    print(f"  Steps     : {result['step_count']}")
    print("─" * 60)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    final_state = run_episode()
    print("\nReasoning chain:")
    for step in final_state["reasoning"]:
        print(f"  {step}")
