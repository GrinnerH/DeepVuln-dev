# app/graph/graph.py
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.core.state import DeepVulnState

# Node implementations (currently stubs; replace internals progressively)
from app.graph.nodes import (
    finalize_and_report,
    ingest_seed_material,
    scan_target_repository,
    setup_environment,
    diagnose_vulnerability_gap,
    synthesize_semantic_model,
    evaluate_modeling_success,
)


def route_after_evaluation(state: DeepVulnState) -> Literal[
    "diagnose_vulnerability_gap",
    "scan_target_repository",
    "finalize_and_report",
]:
    """
    Conditional routing after evaluation:
      - converged => scan target
      - continue  => loop back (bounded by max_iterations)
      - halt/else => finalize
    """
    decision = state.get("active_decision", {}) or {}
    outcome = decision.get("outcome", "halt")

    if outcome == "converged":
        return "scan_target_repository"

    # bounded iteration
    max_iters = state.get("max_iterations", 3)
    iters = state.get("iteration_count", 0)

    if outcome == "continue" and iters < max_iters:
        return "diagnose_vulnerability_gap"

    return "finalize_and_report"


def build_graph():
    """
    DeepVuln 2.0 (flat) topology:
      START -> setup_environment -> ingest_seed_material -> diagnose -> synthesize -> evaluate
                                          ^                                   |
                                          |                                   v
                                          +----------- conditional loop <------+
      converge -> scan_target_repository -> finalize_and_report -> END
    """
    workflow = StateGraph(DeepVulnState)

    # Register nodes
    workflow.add_node("setup_environment", setup_environment)
    workflow.add_node("ingest_seed_material", ingest_seed_material)

    # Explicit modeling loop nodes
    workflow.add_node("diagnose_vulnerability_gap", diagnose_vulnerability_gap)
    workflow.add_node("synthesize_semantic_model", synthesize_semantic_model)
    workflow.add_node("evaluate_modeling_success", evaluate_modeling_success)

    # Post-loop nodes
    workflow.add_node("scan_target_repository", scan_target_repository)
    workflow.add_node("finalize_and_report", finalize_and_report)

    # Static backbone
    workflow.add_edge(START, "setup_environment")
    workflow.add_edge("setup_environment", "ingest_seed_material")
    workflow.add_edge("ingest_seed_material", "diagnose_vulnerability_gap")

    # Loop edges (explicit in the graph)
    workflow.add_edge("diagnose_vulnerability_gap", "synthesize_semantic_model")
    workflow.add_edge("synthesize_semantic_model", "evaluate_modeling_success")

    # Conditional routing after evaluation
    workflow.add_conditional_edges("evaluate_modeling_success", route_after_evaluation)

    # After convergence: scan target then finalize
    workflow.add_edge("scan_target_repository", "finalize_and_report")
    workflow.add_edge("finalize_and_report", END)

    return workflow.compile()
