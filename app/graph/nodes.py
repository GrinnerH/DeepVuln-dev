from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.agent.model import build_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.state import DeepVulnState, EvidenceRef, GapDiagnosis, ModelingDecision, ModelingRecord
from app.mcp.codeql_mcp import CodeQLMCPClient, CodeQLExecutionError
from utils.io import ensure_dir, safe_read_text
from utils.code_window import extract_code_window
from utils.jsonx import ensure_list, json_loads_best_effort
from utils.seed_loader import (
    load_and_normalize_seeds,
    SeedMaterialError,
    bootstrap_seed_case,
    load_seed_cases,
    normalize_seed_case,
    save_seed_cases,
)

logger = logging.getLogger("deepvuln.v2")


# -----------------------------
# Small helpers (keep nodes readable)
# -----------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _init_file_logger(*, logs_dir: Path, run_id: Optional[str] = None) -> None:
    """Attach a file handler to the module logger.

    This keeps execution traces out of the LangGraph state, while still
    providing a durable, per-run audit trail on disk.

    The handler is added idempotently (safe across retries / re-entry).
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id or "run"
    log_path = logs_dir / f"{rid}.trace.log"

    # Avoid duplicate handlers (important in notebook/retry contexts).
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(getattr(h, "baseFilename", "")) == log_path:
                    return
            except Exception:
                continue

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Do not force global basicConfig; keep integration-friendly.
    logger.setLevel(logging.INFO)


def _read_code_window(file_path: Path, line: int, window: int = 25) -> str:
    """
    Returns a formatted code window around `line` with line numbers.
    Falls back gracefully if file is missing/unreadable.
    """
    try:
        content = safe_read_text(str(file_path))
    except Exception as e:
        return f"[source-unavailable] Failed to read {file_path}: {e}"

    lines = content.splitlines()
    if not lines:
        return f"[source-empty] {file_path}"

    # convert to 0-indexed indices
    idx = max(0, line - 1)
    lo = max(0, idx - window)
    hi = min(len(lines), idx + window + 1)

    # format with 1-indexed line numbers
    out = []
    for i in range(lo, hi):
        prefix = ">>" if i == idx else "  "
        out.append(f"{prefix} {i+1:6d}: {lines[i]}")
    return "\n".join(out)


def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s[:180] if len(s) > 180 else s


# -----------------------------
# Stage A: setup + ingest
# -----------------------------


def setup_environment(state: DeepVulnState) -> Dict[str, Any]:
    """Create run workspace directories and initialize control fields.

    Notes:
      - This node returns only minimal, serializable fields for state updates.
      - Runtime logs are written to a local trace file (not stored in state).
    """
    workspace = Path(state.get("workspace_dir") or "./artifacts/run")
    _ensure_dir(workspace)
    _ensure_dir(workspace / "baseline")
    _ensure_dir(workspace / "semantic_packs")
    _ensure_dir(workspace / "reports")
    logs_dir = workspace / "logs"
    _ensure_dir(logs_dir)

    # Initialize a per-run file logger (idempotent).
    _init_file_logger(logs_dir=logs_dir, run_id=state.get("run_id"))

    logger.info("Workspace initialized at %s", workspace)

    return {
        "workspace_dir": str(workspace),
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "is_converged": False,
    }


def ingest_seed_material(state: DeepVulnState) -> Dict[str, Any]:
    """
    Consume seed manifest produced by Step A.

    Priority:
      1) seeds/seed_assets.json (preferred, produced by scripts/bootstrap_seed_assets.py)
      2) seeds/seed_cases.json + bootstrap (dev fallback) -> also writes seed_assets.json

    Note: No verbose logs are stored in state; use local trace file + on-disk artifacts.
    """
    workspace_dir = state.get("workspace_dir") or "./artifacts/run"
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    seed_assets_path = Path("seeds/seed_assets.json")
    seed_cases_path = Path("seeds/seed_cases.json")

    codeql = CodeQLMCPClient()

    # 1) Preferred path: consume manifest
    if seed_assets_path.exists():
        raw = load_seed_cases(str(seed_assets_path))
        normalized = [normalize_seed_case(workspace_dir, s) for s in raw]
        logger.info("Loaded seed manifest: %s (n=%d)", seed_assets_path, len(normalized))
        return {"seed_cases": normalized}

    # 2) Dev fallback: bootstrap then persist manifest
    raw = load_seed_cases(str(seed_cases_path))
    enriched = []
    for s in raw:
        enriched.append(
            bootstrap_seed_case(
                workspace_dir=workspace_dir,
                seed=s,
                codeql=codeql,
                overwrite_db=False,
            )
        )

    save_seed_cases(str(seed_assets_path), enriched)
    normalized = [normalize_seed_case(workspace_dir, s) for s in enriched]
    logger.info(
        "Bootstrapped from %s and wrote seed manifest: %s (n=%d)",
        seed_cases_path,
        seed_assets_path,
        len(normalized),
    )
    return {"seed_cases": normalized}


# -----------------------------
# Stage B loop: diagnose -> synthesize -> evaluate
# -----------------------------


def diagnose_vulnerability_gap(state: DeepVulnState) -> Dict[str, Any]:
    workspace_dir = Path(state["workspace_dir"])
    ensure_dir(workspace_dir)

    seed_cases = state.get("seed_cases", [])
    if not seed_cases:
        return {
            "active_diagnosis": {
                "gap_kind": "unknown",
                "hypothesis": "No seed_cases provided. Cannot diagnose semantic gap.",
                "supporting_evidence_refs": [],
                "docs_citation_refs": [],
                "modeling_plan": [],
                "supporting_evidence": [],
            }
        }

    seed0 = seed_cases[0]
    cve_id = seed0.get("cve_id") or "UNKNOWN_CVE"

    # --- Canonical schema only (NO key-guessing) ---
    repo_path = Path(seed0["repo_path"])
    db_path = seed0["db_path"]
    vuln_file_rel = seed0["vulnerable_file"]
    vuln_line = int(seed0.get("vulnerable_line") or 1)

    baseline_query_path = seed0.get("baseline_query_path") or os.getenv("BASELINE_QUERY_PATH", "")
    codeql_search_paths = seed0.get("codeql_search_paths") or []

    logger.info(
        "[diagnose] cve=%s repo=%s db=%s vuln_file=%s vuln_line=%s",
        cve_id,
        repo_path,
        db_path,
        vuln_file_rel,
        vuln_line,
    )

    diagnostics_dir = ensure_dir(workspace_dir / "diagnostics")
    evidence_dir = ensure_dir(workspace_dir / "evidence")

    baseline_log_path = diagnostics_dir / f"{cve_id}_baseline.json"
    baseline_sarif_path = diagnostics_dir / f"{cve_id}_baseline.sarif"

    # ---- 1) Baseline CodeQL run (database analyze) ----
    baseline_result: Dict[str, Any] = {}
    baseline_summary = "NOT_RUN"

    try:
        codeql = CodeQLMCPClient()

        if baseline_query_path and db_path:
            baseline_result = codeql.analyze_database(
                db_path=str(db_path),
                query_or_suite_path=str(baseline_query_path),
                output_sarif_path=str(baseline_sarif_path),
                additional_search_paths=codeql_search_paths or None,
                extra_args=None,
                cwd=str(repo_path) if repo_path else None,
            )
            baseline_summary = "OK" if baseline_result.get("ok") else "FAILED"
        else:
            baseline_summary = "SKIPPED (missing baseline_query_path or db_path)"
            baseline_result = {"ok": False, "reason": baseline_summary}
    except Exception as e:
        baseline_summary = f"ERROR: {e}"
        baseline_result = {"ok": False, "error": str(e)}

    baseline_log_path.write_text(json.dumps(baseline_result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[diagnose] baseline_summary=%s baseline_log=%s", baseline_summary, baseline_log_path)

    evidence_refs: List[EvidenceRef] = []
    evidence_refs.append(
        {
            "kind": "diagnostic_log",
            "path": str(baseline_log_path),
            "note": f"Baseline CodeQL analyze result: {baseline_summary}",
        }
    )
    if baseline_sarif_path.exists():
        evidence_refs.append(
            {
                "kind": "sarif",
                "path": str(baseline_sarif_path),
                "note": "Baseline SARIF output (may be empty if no alerts).",
            }
        )

    # ---- 2) Vulnerable code window evidence ----
    abs_vuln_file = repo_path / vuln_file_rel
    code_snip = _read_code_window(abs_vuln_file, line=vuln_line, window=25)
    snippet_path = evidence_dir / f"{cve_id}_{_sanitize_filename(vuln_file_rel)}_L{vuln_line}.code.txt"
    snippet_path.write_text(code_snip, encoding="utf-8")

    evidence_refs.append(
        {
            "kind": "code_snippet",
            "path": str(snippet_path),
            "note": "Code window around vulnerable location.",
        }
    )

    # ---- 3) LLM: failure-driven diagnosis ----
    # Prompt constraints: LLM suggests semantics; CodeQL remains the final verifier.
    prompt = _DIAGNOSE_PROMPT_TEMPLATE.format(
        cve_id=cve_id,
        baseline_summary=baseline_summary,
        baseline_log_path=str(baseline_log_path),
        baseline_sarif_path=str(baseline_sarif_path),
        vulnerable_file=vuln_file_rel,
        vulnerable_line=vuln_line,
        code_window=code_snip,
        target_function=seed0.get("target_function") or "",
    )

    runnable = build_chat_model(structured_schema=None)
    messages = [
        SystemMessage(content="You are a security analysis assistant specializing in CodeQL semantic modeling."),
        HumanMessage(content=prompt),
    ]

    raw = runnable.invoke(messages)
    raw_text = getattr(raw, "content", raw)

    parsed = json_loads_best_effort(raw_text, fallback={})
    gap_kind = parsed.get("gap_kind", "unknown")
    hypothesis = parsed.get("hypothesis", "")
    modeling_plan = ensure_list(parsed.get("modeling_plan", []))
    docs_needed = ensure_list(parsed.get("docs_needed", []))
    supporting_evidence = ensure_list(parsed.get("supporting_evidence", []))

    diagnosis: GapDiagnosis = {
        "gap_kind": gap_kind,
        "hypothesis": hypothesis,
        "supporting_evidence_refs": evidence_refs,
        "docs_citation_refs": docs_needed,
        "modeling_plan": modeling_plan,
        "supporting_evidence": supporting_evidence,
    }

    logger.info("[diagnose] gap_kind=%s modeling_plan_items=%d", gap_kind, len(modeling_plan))
    return {"active_diagnosis": diagnosis}


def synthesize_semantic_model(state: DeepVulnState) -> Dict[str, Any]:
    workspace_dir = Path(state["workspace_dir"])
    ensure_dir(workspace_dir)

    seed_cases = state.get("seed_cases", [])
    if not seed_cases:
        return {"active_semantic_artifact": {"semantic_pack_path": ""}}

    seed0 = seed_cases[0]
    cve_id = seed0.get("cve_id") or "UNKNOWN_CVE"

    diagnosis = state.get("active_diagnosis") or {}
    gap_kind = diagnosis.get("gap_kind", "unknown")
    hypothesis = diagnosis.get("hypothesis", "")
    modeling_plan = diagnosis.get("modeling_plan", [])

    iter_idx = int(state.get("iteration_count", 0)) + 1
    pack_dir = ensure_dir(workspace_dir / "semantic_packs" / f"iter_{iter_idx:02d}")
    qll_path = pack_dir / "ProjectSemantics.qll"

    prompt = _SYNTHESIZE_PROMPT_TEMPLATE.format(
        cve_id=cve_id,
        gap_kind=gap_kind,
        hypothesis=hypothesis,
        modeling_plan="\n".join([f"- {x}" for x in modeling_plan]) if modeling_plan else "(none)",
    )

    runnable = build_chat_model(structured_schema=None)
    messages = [
        SystemMessage(content="You generate safe, minimal CodeQL semantic packs for C/C++ projects."),
        HumanMessage(content=prompt),
    ]
    raw = runnable.invoke(messages)
    raw_text = getattr(raw, "content", raw)

    # The LLM should output CodeQL (.qll) content.
    qll_path.write_text(str(raw_text), encoding="utf-8")
    logger.info("[synthesize] wrote semantic pack: %s", qll_path)

    artifact = {
        "semantic_pack_path": str(qll_path),
    }
    return {"active_semantic_artifact": artifact}


def evaluate_modeling_success(state: DeepVulnState) -> Dict[str, Any]:
    workspace_dir = Path(state["workspace_dir"])
    ensure_dir(workspace_dir)

    seed_cases = state.get("seed_cases", [])
    if not seed_cases:
        decision = {"outcome": "halt", "rationale": "No seed_cases; cannot evaluate modeling success."}
        return {"active_decision": decision, "is_converged": False}

    seed0 = seed_cases[0]
    cve_id = seed0.get("cve_id") or "UNKNOWN_CVE"

    # --- Canonical schema only (NO key-guessing) ---
    db_path = seed0["db_path"]
    vuln_file_rel = seed0["vulnerable_file"]
    vuln_line = int(seed0.get("vulnerable_line") or 1)
    codeql_search_paths = seed0.get("codeql_search_paths") or []

    artifact = state.get("active_semantic_artifact") or {}
    qll_path = artifact.get("semantic_pack_path")
    if not qll_path:
        decision = {"outcome": "halt", "rationale": "No active semantic artifact path to compile/analyze."}
        return {"active_decision": decision, "is_converged": False}

    qll_path_p = Path(qll_path)
    iter_idx = int(state.get("iteration_count", 0)) + 1

    diagnostics_dir = ensure_dir(workspace_dir / "diagnostics")
    compile_log_path = diagnostics_dir / f"{cve_id}_compile_iter_{iter_idx:02d}.json"
    regression_sarif_path = diagnostics_dir / f"{cve_id}_regression_iter_{iter_idx:02d}.sarif"
    regression_log_path = diagnostics_dir / f"{cve_id}_regression_iter_{iter_idx:02d}.json"

    # Local client (do NOT store in state; keep state serializable)
    codeql = CodeQLMCPClient()

    # ---- 1) Compile the generated semantic pack (query compile) ----
    compile_res = codeql.query_compile(
        query_path=str(qll_path_p),
        additional_search_paths=codeql_search_paths or None,
        extra_args=None,
        cwd=None,
    )
    compile_log_path.write_text(json.dumps(compile_res, ensure_ascii=False, indent=2), encoding="utf-8")

    compile_ok = bool(compile_res.get("ok"))
    logger.info("[evaluate] compile_ok=%s compile_log=%s", compile_ok, compile_log_path)

    artifact_updated = dict(artifact)
    artifact_updated["compile_log_ref"] = {
        "kind": "diagnostic_log",
        "path": str(compile_log_path),
        "note": "CodeQL query compile output for semantic pack.",
    }

    if not compile_ok:
        decision: ModelingDecision = {
            "outcome": "continue",
            "rationale": "Semantic pack failed to compile; refine semantics and retry.",
        }
        record: ModelingRecord = {
            "iteration_index": iter_idx,
            "diagnosis": state.get("active_diagnosis") or {},
            "semantic_artifact": artifact_updated,
            "metrics": {"compile_ok": False, "seed_hit_rate": 0.0},
            "decision": decision,
        }
        return {
            "active_semantic_artifact": artifact_updated,
            "active_decision": decision,
            "modeling_history": [record],
            "iteration_count": iter_idx,
            "is_converged": False,
        }

    # ---- 2) Regression run on the seed DB ----
    regression_res = codeql.analyze_database(
        db_path=str(db_path),
        query_or_suite_path=str(qll_path_p),
        output_sarif_path=str(regression_sarif_path),
        additional_search_paths=codeql_search_paths or None,
        extra_args=None,
        cwd=None,
    )
    regression_log_path.write_text(json.dumps(regression_res, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[evaluate] regression_ok=%s sarif=%s", regression_res.get("ok"), regression_sarif_path)

    # ---- 3) Compute seed hit rate (proxy oracle) ----
    locs = _parse_sarif_locations(regression_sarif_path) if regression_sarif_path.exists() else []
    hit = _hit_vuln_site(locs, vuln_file_rel=vuln_file_rel, vuln_line=vuln_line, radius=3)
    seed_hit_rate = 1.0 if hit else 0.0

    outcome: ModelingDecision["outcome"]
    if hit:
        outcome = "converged"
        rationale = "Regression run produced a SARIF location near the vulnerable site."
        best_pack = str(qll_path_p)
        is_converged = True
    else:
        if iter_idx >= int(state.get("max_iterations", 3)):
            outcome = "halt"
            rationale = "Max iterations reached without hitting seed site."
            best_pack = state.get("best_semantic_pack_path", "")
            is_converged = False
        else:
            outcome = "continue"
            rationale = "No SARIF hit near seed site; refine semantics and retry."
            best_pack = state.get("best_semantic_pack_path", "")
            is_converged = False

    decision = {"outcome": outcome, "rationale": rationale}

    record: ModelingRecord = {
        "iteration_index": iter_idx,
        "diagnosis": state.get("active_diagnosis") or {},
        "semantic_artifact": artifact_updated,
        "metrics": {"compile_ok": True, "seed_hit_rate": seed_hit_rate},
        "decision": decision,
    }

    updates: Dict[str, Any] = {
        "active_semantic_artifact": artifact_updated,
        "active_decision": decision,
        "modeling_history": [record],
        "iteration_count": iter_idx,
        "is_converged": is_converged,
    }
    if best_pack:
        updates["best_semantic_pack_path"] = best_pack

    return updates


# -----------------------------
# Utility: SARIF parsing / hit check
# -----------------------------


def _parse_sarif_locations(sarif_path: Path) -> List[Dict[str, Any]]:
    """
    Extract a flat list of locations from SARIF results.
    Each item: { uri, startLine, endLine, ruleId, message }
    """
    try:
        sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    runs = sarif.get("runs") or []
    for run in runs:
        results = run.get("results") or []
        for r in results:
            rule_id = r.get("ruleId")
            msg = (r.get("message") or {}).get("text", "")
            locs = (r.get("locations") or [])
            for loc in locs:
                pl = (loc.get("physicalLocation") or {})
                art = (pl.get("artifactLocation") or {})
                reg = (pl.get("region") or {})
                uri = art.get("uri")
                out.append(
                    {
                        "uri": uri,
                        "startLine": reg.get("startLine"),
                        "endLine": reg.get("endLine"),
                        "ruleId": rule_id,
                        "message": msg,
                    }
                )
    return out


def _hit_vuln_site(locs: List[Dict[str, Any]], vuln_file_rel: str, vuln_line: int, radius: int = 3) -> bool:
    """
    Proxy oracle for V1:
    If any SARIF location matches the vulnerable file and within +/- radius lines, count as hit.
    """
    if not vuln_file_rel:
        return False

    vf_norm = vuln_file_rel.replace("\\", "/")
    for it in locs:
        uri = str(it.get("uri", "")).replace("\\", "/")
        if uri.endswith(vf_norm):
            sl = it.get("startLine")
            if isinstance(sl, int) and abs(sl - vuln_line) <= radius:
                return True
    return False


# -----------------------------
# Prompts
# -----------------------------


_DIAGNOSE_PROMPT_TEMPLATE = r"""
You are given a real-world seed vulnerability exemplar.

Your job is NOT to decide the vulnerability, but to diagnose why baseline CodeQL did NOT flag it (or was inconclusive),
and identify what project-specific semantics are missing.

Constraints:
- You must be failure-driven: use baseline SARIF/logs and the code window as evidence.
- You must propose testable hypotheses and a concrete semantic modeling plan.
- You must NOT invent CodeQL APIs. If you need docs, request them via docs_needed.

Inputs:
- CVE: {cve_id}
- Baseline summary: {baseline_summary}
- Baseline log path: {baseline_log_path}
- Baseline SARIF path: {baseline_sarif_path}

Vulnerable site anchor:
- File: {vulnerable_file}
- Line: {vulnerable_line}
- Target function: {target_function}

Code window:
{code_window}

Required output:
1) Pick one gap_kind from:
   - unknown
   - sink_semantics_missing
   - capacity_semantics_missing
   - length_semantics_missing
   - guard_semantics_missing
   - logic_mismatch

2) Provide a falsifiable hypothesis:
   - Must reference code details (fields, macros, wrappers, guards).
   - Must explain why baseline CodeQL cannot prove overflow without additional semantics.

3) Provide a minimal modeling_plan (2-5 items):
   - Each item should describe an implementable semantic hook in a ProjectSemantics.qll pack
   - Examples: model wrapper as sink, infer capacity from macro/field, treat field as taint source, etc.

4) List additional evidence artifacts that would strengthen the diagnosis (do NOT invent CodeQL APIs).
   - If you need CodeQL docs, request them explicitly by symbol/topic.

Output STRICT JSON only (no markdown):
{{
  "gap_kind": "...",
  "hypothesis": "...",
  "modeling_plan": ["...", "..."],
  "supporting_evidence": [
    "..."
  ],
  "docs_needed": [
    "..."
  ]
}}
"""


_SYNTHESIZE_PROMPT_TEMPLATE = r"""
You are generating a minimal, compiling CodeQL semantic pack for a C/C++ project.

Goal:
- Implement project-specific semantics that close the diagnosed gap.
- Output must be CodeQL (.qll) ONLY.
- Keep it minimal and safe: avoid over-approximation that would explode false positives.

Context:
- CVE: {cve_id}
- gap_kind: {gap_kind}
- hypothesis: {hypothesis}
- modeling_plan:
{modeling_plan}

Rules:
- You may import `cpp` and standard CodeQL C++ libraries.
- You should define predicates / classes that can be used by existing queries.
- Prefer additive modeling (do not rewrite core libraries).
- Do NOT output markdown.

Output: CodeQL (.qll) content only.
"""
