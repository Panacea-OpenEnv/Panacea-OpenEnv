"""
Project Panacea — LangGraph Orchestrator (MongoDB V2)

Assembles the stateful episode graph that runs one full verification cycle:
  load claim → ghost check → primary verify (vitals) → comorbidity deep-check
  → analyze → submit → reward → telemetry → END
"""

from langgraph.graph import StateGraph, END

from .state import PanaceaState, initial_state
from .nodes import (
    load_pending_claim,
    check_ghost_patient,
    run_primary_verification,
    check_comorbidities,
    analyze_and_decide,
    submit_decision,
    compute_reward,
    log_telemetry,
)

# Conditional edge functions

def route_after_ghost_check(state: PanaceaState) -> str:
    if state["ghost_patient"] or state["verification_status"] == "cannot_verify":
        return "analyze_and_decide"
    return "run_primary_verification"

def route_after_primary_verify(state: PanaceaState) -> str:
    if state.get("verification_status") == "cannot_verify":
        return "analyze_and_decide"
    return "check_comorbidities"

# Graph Assembly

def build_graph() -> StateGraph:
    graph = StateGraph(PanaceaState)

    # Register all nodes
    graph.add_node("load_pending_claim",       load_pending_claim)
    graph.add_node("check_ghost_patient",      check_ghost_patient)
    graph.add_node("run_primary_verification", run_primary_verification)
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

    # Conditional: primary verify result
    graph.add_conditional_edges(
        "run_primary_verification",
        route_after_primary_verify,
        {
            "check_comorbidities":       "check_comorbidities",
            "analyze_and_decide":        "analyze_and_decide",
        },
    )

    # Fixed tail
    graph.add_edge("check_comorbidities",      "analyze_and_decide")
    graph.add_edge("analyze_and_decide",       "submit_decision")
    graph.add_edge("submit_decision",          "compute_reward")
    graph.add_edge("compute_reward",           "log_telemetry")
    graph.add_edge("log_telemetry",            END)

    return graph


# Compiled application
app = build_graph().compile()


# Episode runner helper (used by training loop and demo)

def run_episode(override_state: dict | None = None) -> PanaceaState:
    state = initial_state()
    if override_state:
        state.update(override_state)

    print("\n" + "═" * 60)
    print("  PROJECT PANACEA — LangGraph Episode Starting (MongoDB)")
    print("═" * 60)

    result = app.invoke(state)

    print("\n" + "─" * 60)
    print(f"  Decision  : {result['final_decision']}")
    print(f"  Deception : {result['deception_type']} (detected={result['deception_detected']})")
    print(f"  Reward    : {result['reward']}")
    print(f"  Steps     : {result['step_count']}")
    print("─" * 60)

    return result

if __name__ == "__main__":
    final_state = run_episode()
    print("\nReasoning chain:")
    for step in final_state["reasoning"]:
        print(f"  {step}")
