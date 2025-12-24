# app/core/state.py
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
import operator

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# ----------------------------
# Evidence & diagnostics
# ----------------------------

class EvidenceRef(TypedDict, total=False):
    """
    Lightweight pointer to artifacts produced by CodeQL / tooling.

    Keep raw code, SARIF, and logs on disk; only store pointers here.
    """
    kind: Literal["ast_slice", "cfg_slice", "sarif", "diagnostic_log", "code_snippet", "other"]
    path: str
    note: str


class GapDiagnosis(TypedDict, total=False):
    """
    Failure-driven diagnosis of why baseline detection missed / was inconclusive.
    """
    gap_kind: Literal[
        "unknown",
        "sink_semantics_missing",
        "capacity_semantics_missing",
        "length_semantics_missing",
        "guard_semantics_missing",
        "logic_mismatch",
    ]
    hypothesis: str
    supporting_evidence_refs: List[EvidenceRef]

    # Optional: when nodes consult RAGFlow MCP (CodeQL docs), store citations/refs here.
    docs_citation_refs: List[str]
    learned_patterns: List[Dict[str, Any]]


class SemanticArtifact(TypedDict, total=False):
    """
    The evolving semantic pack (typically .qll) produced during the modeling loop.
    """
    semantic_pack_path: str
    compile_log_ref: EvidenceRef


class ValidationMetrics(TypedDict, total=False):
    """
    Minimal metrics for A + V1 strategy.
    """
    compile_ok: bool
    seed_hit_rate: float

    # V1 robustness metrics (optional in MVP; keep fields for later expansion).
    lexical_robustness: float
    fp_rate: float


class ModelingDecision(TypedDict, total=False):
    """
    Drives graph routing after evaluation.
    """
    outcome: Literal["continue", "converged", "halt"]
    rationale: str


class ModelingRecord(TypedDict, total=False):
    """
    A single iteration snapshot (for paper-grade auditability).
    """
    iteration_index: int
    diagnosis: GapDiagnosis
    semantic_artifact: SemanticArtifact
    metrics: ValidationMetrics
    decision: ModelingDecision


# ----------------------------
# Global State (LangGraph)
# ----------------------------

class DeepVulnState(TypedDict, total=False):
    # Run context
    run_id: str
    target_cwe_id: str
    workspace_dir: str
    target_repo_path: str

    # Conversation backbone (optional; useful if you later attach chat models/tools)
    messages: Annotated[List[AnyMessage], add_messages]

    # Seed material
    seed_cases: List[Dict[str, Any]]

    # Loop control
    iteration_count: int
    max_iterations: int
    is_converged: bool

    # Node-to-node active handoff (ephemeral)
    active_diagnosis: GapDiagnosis
    active_semantic_artifact: SemanticArtifact
    active_decision: ModelingDecision

    # Modeling history (auto-appended via reducer)
    modeling_history: Annotated[List[ModelingRecord], operator.add]

    # Final deliverables
    best_semantic_pack_path: str
    scanning_result: Dict[str, Any]
    error_log: List[str]
