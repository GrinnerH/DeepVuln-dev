from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.llm.model import build_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.state import DeepVulnState, EvidenceRef, GapDiagnosis, ModelingDecision, ModelingRecord
from app.mcp.codeql_mcp import CodeQLMCPClient, CodeQLExecutionError
from app.utils.io import ensure_dir, safe_read_text
from app.utils.code_window import extract_code_window
from app.utils.jsonx import ensure_list, json_loads_best_effort
from app.utils.seed_loader import load_and_normalize_seeds, SeedMaterialError
logger = logging.getLogger("deepvuln.v2")


# -----------------------------
# Small helpers (keep nodes readable)
# -----------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_source_window(repo_path: str, rel_file: str, line: int, window: int = 80) -> str:
    """Read a window of source lines around a 1-indexed line."""
    file_path = Path(repo_path) / rel_file
    if not file_path.exists():
        return f"[source-unavailable] File not found: {file_path}"
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"[source-unavailable] Failed to read {file_path}: {e}"

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


def _normalize_seed_case(seed: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort normalization to a canonical shape used by nodes."""
    # Common aliases observed in earlier notes / seeds files.
    repo_path = seed.get("repo_path") or seed.get("local_repo_path") or seed.get("repo")
    db_path = seed.get("codeql_db_path") or seed.get("db_path") or seed.get("database")
    vuln_file = seed.get("vuln_file") or seed.get("vulnerable_file") or seed.get("file")
    vuln_line = seed.get("vuln_line") or seed.get("line") or seed.get("vulnerable_line")

    # If only absolute file path is provided, try to convert to repo-relative.
    if vuln_file and repo_path and os.path.isabs(vuln_file):
        try:
            vuln_file = str(Path(vuln_file).resolve().relative_to(Path(repo_path).resolve()))
        except Exception:
            # keep as-is
            pass

    out = dict(seed)
    if repo_path:
        out["repo_path"] = repo_path
    if db_path:
        out["codeql_db_path"] = db_path
    if vuln_file:
        out["vuln_file"] = vuln_file
    if vuln_line is not None:
        try:
            out["vuln_line"] = int(vuln_line)
        except Exception:
            out["vuln_line"] = vuln_line
    return out


def _sarif_has_hit_near_location(
    sarif_path: str, repo_path: str, rel_file: str, vuln_line: int, tolerance: int = 3
) -> bool:
    """Best-effort hit check: any SARIF result location within +/-tolerance lines."""
    try:
        data = json.loads(Path(sarif_path).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    runs = data.get("runs") or []
    target_path = str(Path(repo_path) / rel_file)
    target_path_norm = os.path.normpath(target_path)

    for run in runs:
        results = run.get("results") or []
        for r in results:
            locs = (r.get("locations") or [])
            for loc in locs:
                ploc = (loc.get("physicalLocation") or {})
                artifact = (ploc.get("artifactLocation") or {})
                uri = artifact.get("uri") or ""
                region = (ploc.get("region") or {})
                start_line = region.get("startLine")
                if start_line is None:
                    continue

                # resolve SARIF uri to an absolute-ish path if possible.
                if uri.startswith("file:"):
                    uri_path = uri.replace("file:", "")
                else:
                    uri_path = uri
                uri_path_norm = os.path.normpath(os.path.join(repo_path, uri_path))

                # Compare normalized paths.
                if uri_path_norm != target_path_norm:
                    continue
                try:
                    if abs(int(start_line) - int(vuln_line)) <= tolerance:
                        return True
                except Exception:
                    continue
    return False


# -----------------------------
# Non-loop nodes (stubs remain lightweight)
# -----------------------------


def setup_environment(state: DeepVulnState) -> Dict[str, Any]:
    """Create run workspace directories and initialize control fields."""
    workspace = Path(state.get("workspace_dir") or "./artifacts/run")
    _ensure_dir(workspace)
    _ensure_dir(workspace / "baseline")
    _ensure_dir(workspace / "semantic_packs")
    _ensure_dir(workspace / "reports")
    _ensure_dir(workspace / "logs")

    return {
        "workspace_dir": str(workspace),
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "is_converged": False,
        "error_log": state.get("error_log", []),
    }


def ingest_seed_material(state: DeepVulnState) -> Dict[str, Any]:
    """
    Load and normalize seed cases into DeepVulnState.seed_cases.

    V1 philosophy:
    - Do NOT auto-clone or auto-create CodeQL DB here (to keep ingestion deterministic).
    - Do validate existence and fail fast with actionable errors.
    """
    workspace_dir = state.get("workspace_dir")
    if not workspace_dir:
        raise RuntimeError("workspace_dir is missing. Ensure setup_environment sets it.")

    try:
        seed_cases: List[Dict[str, Any]] = load_and_normalize_seeds(workspace_dir)
    except SeedMaterialError as e:
        logger.exception("Seed material load failed")
        return {
            "error_log": [f"[ingest_seed_material] {e}"],
        }

    errors: List[str] = []
    for s in seed_cases:
        repo_path = Path(s["repo_path"])
        db_path = Path(s["db_path"])

        if not repo_path.exists():
            errors.append(
                f"[seed:{s['cve_id']}] repo_path not found: {repo_path}. "
                f"Please clone {s['repo_url']} into that path, or add repo_path to seed_cases.json."
            )
        else:
            vuln_file = repo_path / s["vulnerable_file"]
            if not vuln_file.exists():
                errors.append(
                    f"[seed:{s['cve_id']}] vulnerable_file not found: {vuln_file}. "
                    f"Check vulnerable_file path in seed_cases.json."
                )

        if not db_path.exists():
            # V1: allow missing DB, but make it explicit (diagnose/evaluate will SKIP analyze)
            errors.append(
                f"[seed:{s['cve_id']}] codeql db_path not found: {db_path}. "
                f"V1 will not auto-create DB. Create it before running baseline/regression analyze."
            )

        # Baseline query/suite is optional but recommended
        if not s.get("baseline_query_path"):
            errors.append(
                f"[seed:{s['cve_id']}] baseline_query_path is not set. "
                f"Set env BASELINE_QUERY_PATH or create queries/baseline.qls."
            )
        else:
            q = Path(s["baseline_query_path"])
            if not q.exists():
                errors.append(
                    f"[seed:{s['cve_id']}] baseline_query_path does not exist: {q}. "
                    f"Fix BASELINE_QUERY_PATH or queries/baseline.qls."
                )

    update: Dict[str, Any] = {
        "seed_cases": seed_cases,
    }

    if errors:
        # append to error_log (ensure reducer/append semantics in your state if desired)
        update["error_log"] = [f"[ingest_seed_material] {msg}" for msg in errors]

    logger.info("Loaded %d seed case(s).", len(seed_cases))
    return update


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
    cve_id = seed0.get("cve_id") or seed0.get("id") or "UNKNOWN_CVE"
    repo_path = Path(seed0.get("repo_path", "") or "")
    db_path = seed0.get("db_path", "")
    vuln_file_rel = seed0.get("vulnerable_file") or seed0.get("file_path") or seed0.get("vuln_file")
    vuln_line = int(seed0.get("line_number") or seed0.get("vuln_line") or 1)

    baseline_query_path = seed0.get("baseline_query_path") or os.getenv("BASELINE_QUERY_PATH", "")
    codeql_search_paths = seed0.get("codeql_search_paths") or []

    diagnostics_dir = ensure_dir(workspace_dir / "diagnostics")
    evidence_dir = ensure_dir(workspace_dir / "evidence")

    baseline_log_path = diagnostics_dir / f"{cve_id}_baseline.json"
    baseline_sarif_path = diagnostics_dir / f"{cve_id}_baseline.sarif"

    # ---- 1) Baseline CodeQL run (database analyze) ----
    baseline_result: Dict[str, Any] = {}
    baseline_summary = "NOT_RUN"

    try:
        codeql = state.get("codeql_client")
        if codeql is None:
            from app.codeql.client import CodeQLMCPClient  # adjust to your actual module path
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

    baseline_log_path.write_text(
        json.dumps(
            {
                "cve_id": cve_id,
                "db_path": db_path,
                "baseline_query_path": baseline_query_path,
                "baseline_summary": baseline_summary,
                "baseline_result": baseline_result,
                "baseline_sarif_path": str(baseline_sarif_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- 2) Read code slice around vuln line ----
    code_slice_path = evidence_dir / f"{cve_id}_code_slice.txt"
    file_abs_path = (repo_path / vuln_file_rel).resolve() if repo_path and vuln_file_rel else None

    if file_abs_path and file_abs_path.exists():
        src_text = safe_read_text(file_abs_path)
        window_info = extract_code_window(src_text, vuln_line, window=30)
    else:
        window_info = {
            "start_line": 1,
            "end_line": 1,
            "snippet": f"<<MISSING_FILE repo_path={repo_path} vuln_file={vuln_file_rel}>>",
        }

    code_slice_path.write_text(
        f"# file: {vuln_file_rel}\n# abs: {file_abs_path}\n# focus_line: {vuln_line}\n\n{window_info['snippet']}\n",
        encoding="utf-8",
    )

    # ---- 3) Failure-driven prompt ----
    prompt = f"""
You are a security researcher specializing in CodeQL-based vulnerability analysis and semantic gap diagnosis.

We follow a failure-driven workflow:
- Run baseline CodeQL on a known vulnerable seed.
- Diagnose which missing semantic knowledge caused the failure (or weak proof).
- Produce a falsifiable hypothesis and a minimal semantic-pack plan.

Seed case:
- CVE: {cve_id}
- Repo path (local): {str(repo_path)}
- Vulnerable file: {vuln_file_rel}
- Vulnerable line: {vuln_line}

Baseline CodeQL run summary:
- Status: {baseline_summary}
- codeql stderr (truncated):
{(baseline_result.get("stderr") or "")[:2000]}
- SARIF path (if produced): {str(baseline_sarif_path)}

Vulnerable code slice ('>>' marks focal line):
{window_info["snippet"]}

Your task:
1) Choose exactly ONE gap_kind from:
   - "unknown"
   - "sink_semantics_missing"
   - "capacity_semantics_missing"
   - "length_semantics_missing"
   - "guard_semantics_missing"
   - "logic_mismatch"

2) Provide a falsifiable hypothesis:
   - Must reference concrete program elements visible in the slice (identifiers, fields, functions, macros).
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
    {{"kind": "ast_slice|cfg_slice|sarif|diagnostic_log|other", "note": "what to capture next"}}
  ],
  "docs_needed": ["<exact CodeQL symbol/topic to look up>"]
}}

Constraints:
- Do NOT invent CodeQL APIs/predicates. If uncertain, put it in docs_needed.
""".strip()

    model = build_chat_model(
        model_name=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        tools=[],               # later bind RAGFlow MCP tools optionally
        structured_schema=None  # later switch to with_structured_output
    )

    resp = model.invoke(
        [
            SystemMessage(content="Return JSON only; no markdown; no extra text."),
            HumanMessage(content=prompt),
        ]
    )
    llm_text = getattr(resp, "content", None) or str(resp)
    parsed = json_loads_best_effort(llm_text) or {}

    if not parsed:
        parsed = {
            "gap_kind": "unknown",
            "hypothesis": "LLM output was not parseable as JSON.",
            "modeling_plan": [],
            "supporting_evidence": [],
            "docs_needed": [],
        }

    diagnosis = {
        "gap_kind": parsed.get("gap_kind", "unknown"),
        "hypothesis": parsed.get("hypothesis", ""),
        "supporting_evidence_refs": [
            {"kind": "diagnostic_log", "path": str(baseline_log_path), "note": "baseline run result"},
            {"kind": "sarif", "path": str(baseline_sarif_path), "note": "baseline SARIF output"},
            {"kind": "other", "path": str(code_slice_path), "note": "code slice around vulnerable line"},
        ],
        "docs_citation_refs": ensure_list(parsed.get("docs_needed")),
        "modeling_plan": ensure_list(parsed.get("modeling_plan")),
        "supporting_evidence": ensure_list(parsed.get("supporting_evidence")),
    }

    return {"active_diagnosis": diagnosis}


def synthesize_semantic_model(state: DeepVulnState) -> Dict[str, Any]:
    workspace_dir = Path(state["workspace_dir"])
    ensure_dir(workspace_dir)

    seed_cases = state.get("seed_cases", [])
    seed0 = seed_cases[0] if seed_cases else {}
    cve_id = seed0.get("cve_id") or seed0.get("id") or "UNKNOWN_CVE"

    diagnosis = state.get("active_diagnosis") or {}
    gap_kind = diagnosis.get("gap_kind", "unknown")
    hypothesis = diagnosis.get("hypothesis", "")
    modeling_plan = diagnosis.get("modeling_plan", [])

    it_next = int(state.get("iteration_count", 0) or 0) + 1
    pack_dir = ensure_dir(workspace_dir / "semantic_packs" / f"iter_{it_next:02d}")
    out_qll_path = pack_dir / "ProjectSemantics.qll"

    prompt = f"""
You are an expert CodeQL engineer for C/C++ security queries.

Goal:
Generate a single CodeQL QLL file named "ProjectSemantics.qll" that fills a semantic gap revealed by a failure-driven diagnosis.

Seed CVE: {cve_id}

Diagnosis:
- gap_kind: {gap_kind}
- hypothesis: {hypothesis}

Modeling plan:
{modeling_plan}

Output requirements:
1) Output ONLY raw QLL content (no markdown / fences).
2) Must be syntactically valid CodeQL and start with `import cpp` if needed.
3) Keep the semantic pack minimal and focused.
4) Prefer structural modeling over fragile string matching.
5) Include a short header comment explaining:
   - what gap it addresses
   - what hooks it adds
   - assumptions/limits

Strict constraint:
- Do NOT invent CodeQL APIs/predicates. If unsure, insert TODO comments and rely on docs later.

Deliver exactly the QLL file content.
""".strip()

    model = build_chat_model(
        model_name=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        tools=[],               # later bind RAGFlow MCP optionally
        structured_schema=None
    )

    resp = model.invoke(
        [
            SystemMessage(content="Return only valid QLL file content. No markdown."),
            HumanMessage(content=prompt),
        ]
    )
    qll_text = (getattr(resp, "content", None) or str(resp)).strip()

    if not qll_text:
        qll_text = (
            "/**\n"
            " * Auto-generated ProjectSemantics.qll (EMPTY FALLBACK)\n"
            " * NOTE: LLM returned empty output.\n"
            " */\n"
            "import cpp\n\n"
            "// TODO: implement semantic hooks\n"
        )

    out_qll_path.write_text(qll_text + "\n", encoding="utf-8")

    artifact = {
        "semantic_pack_path": str(out_qll_path),
        "compile_log_ref": {"kind": "diagnostic_log", "path": "", "note": "filled in evaluate_modeling_success"},
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
    cve_id = seed0.get("cve_id") or seed0.get("id") or "UNKNOWN_CVE"
    db_path = seed0.get("db_path", "")
    vuln_file_rel = seed0.get("vulnerable_file") or seed0.get("file_path") or seed0.get("vuln_file") or ""
    vuln_line = int(seed0.get("line_number") or seed0.get("vuln_line") or 1)

    codeql_search_paths = seed0.get("codeql_search_paths") or []

    artifact = state.get("active_semantic_artifact") or {}
    qll_path = artifact.get("semantic_pack_path")
    if not qll_path:
        decision = {"outcome": "halt", "rationale": "No active semantic artifact path to compile/analyze."}
        return {"active_decision": decision, "is_converged": False}

    qll_path_p = Path(qll_path)

    diagnostics_dir = ensure_dir(workspace_dir / "diagnostics")
    compile_log_path = diagnostics_dir / f"{cve_id}_compile_iter_{int(state.get('iteration_count', 0))+1:02d}.json"

    regression_sarif_path = diagnostics_dir / f"{cve_id}_regression_iter_{int(state.get('iteration_count', 0))+1:02d}.sarif"
    regression_log_path = diagnostics_dir / f"{cve_id}_regression_iter_{int(state.get('iteration_count', 0))+1:02d}.json"

    # Get codeql client
    codeql = state.get("codeql_client")
    if codeql is None:
        from app.codeql.client import CodeQLMCPClient  # adjust if needed
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

    # Update active_semantic_artifact.compile_log_ref
    artifact_updated = dict(artifact)
    artifact_updated["compile_log_ref"] = {
        "kind": "diagnostic_log",
        "path": str(compile_log_path),
        "note": "codeql query compile log",
    }

    # Early decision if compile fails
    it_next = int(state.get("iteration_count", 0) or 0) + 1
    max_iter = int(state.get("max_iterations", 3) or 3)

    if not compile_ok:
        decision = {
            "outcome": "continue" if it_next < max_iter else "halt",
            "rationale": "Semantic pack failed to compile; see compile log and refine.",
        }
        record: ModelingRecord = {
            "iteration_index": it_next,
            "diagnosis": state.get("active_diagnosis", {}),
            "semantic_artifact": artifact_updated,
            "metrics": {"compile_ok": False, "seed_hit_rate": 0.0},
            "decision": decision,
        }
        return {
            "active_semantic_artifact": artifact_updated,
            "modeling_history": [record],
            "iteration_count": it_next,
            "is_converged": False,
            "active_decision": decision,
        }

    # ---- 2) Regression analyze on seed db ----
    # V1: if seed provides regression_query_path use it; else fallback to baseline_query_path; else analyze the qll itself
    regression_query = (
        seed0.get("regression_query_path")
        or seed0.get("baseline_query_path")
        or str(qll_path_p)
    )

    regression_res = codeql.analyze_database(
        db_path=str(db_path),
        query_or_suite_path=str(regression_query),
        output_sarif_path=str(regression_sarif_path),
        additional_search_paths=codeql_search_paths or None,
        extra_args=None,
        cwd=None,
    )
    regression_log_path.write_text(json.dumps(regression_res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 3) Compute hit_rate (V1 proxy oracle) ----
    locs = _parse_sarif_locations(regression_sarif_path)
    hit = _hit_vuln_site(locs, vuln_file_rel=vuln_file_rel, vuln_line=vuln_line, radius=8)

    seed_hit_rate = 1.0 if hit else 0.0

    # ---- 4) Convergence decision ----
    # V1 default: converge if hit_rate == 1.0
    if seed_hit_rate >= 1.0:
        decision = {"outcome": "converged", "rationale": "Regression SARIF hits the vulnerable site in seed."}
        is_converged = True
        best_pack = str(qll_path_p)
    else:
        decision = {
            "outcome": "continue" if it_next < max_iter else "halt",
            "rationale": "Regression did not hit vulnerable site; refine semantic model.",
        }
        is_converged = False
        best_pack = state.get("best_semantic_pack_path", "")

    record = {
        "iteration_index": it_next,
        "diagnosis": state.get("active_diagnosis", {}),
        "semantic_artifact": artifact_updated,
        "metrics": {
            "compile_ok": True,
            "seed_hit_rate": seed_hit_rate,
            # V1 placeholders (你后续接入变体对抗性时再补)
            "lexical_robustness": 0.0,
            "fp_rate": 0.0,
        },
        "decision": decision,
    }

    out: Dict[str, Any] = {
        "active_semantic_artifact": artifact_updated,
        "modeling_history": [record],
        "iteration_count": it_next,
        "is_converged": is_converged,
        "active_decision": decision,
    }
    if is_converged:
        out["best_semantic_pack_path"] = best_pack

    return out
# -----------------------------
# Remaining pipeline nodes (simple stubs)
# -----------------------------


def scan_target_repository(state: DeepVulnState) -> Dict[str, Any]:
    logger.info("Scanning target repository (stub)...")
    return {"scanning_result": {"status": "stub"}}


def finalize_and_report(state: DeepVulnState) -> Dict[str, Any]:
    logger.info("Finalizing report (stub)...")
    return {}


# -----------------------------
# Prompt templates (explicit in-code as requested)
# -----------------------------


_GAP_DIAGNOSIS_SCHEMA_HINT = {
    "type": "object",
    "properties": {
        "gap_kind": {
            "type": "string",
            "enum": [
                "unknown",
                "sink_semantics_missing",
                "capacity_semantics_missing",
                "length_semantics_missing",
                "guard_semantics_missing",
                "logic_mismatch",
            ],
        },
        "hypothesis": {"type": "string"},
        "supporting_evidence_refs": {"type": "array", "items": {"type": "object"}},
        "docs_citation_refs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["gap_kind", "hypothesis"],
}


def _build_gap_diagnosis_prompt(
    target_cwe_id: Optional[str],
    baseline_outputs: List[Dict[str, Any]],
    snippet_blocks: List[str],
    evidence_refs: List[EvidenceRef],
) -> str:
    """Failure-driven diagnosis prompt.

    Emphasizes:
    - Start from baseline failure mode (miss/hit + errors).
    - Use *structured evidence* (SARIF/log refs), not hallucinated facts.
    - Produce falsifiable hypothesis that can be tested by changing semantics and re-running.
    """
    baseline_digest = json.dumps(
        [
            {
                "seed_index": o.get("seed_index"),
                "cve_id": o.get("cve_id"),
                "baseline_ok": o.get("baseline_ok"),
                "baseline_hit": o.get("baseline_hit"),
                "baseline_error": o.get("baseline_error"),
                "sarif_path": o.get("sarif_path"),
            }
            for o in baseline_outputs
        ],
        ensure_ascii=False,
        indent=2,
    )

    evidence_digest = json.dumps(evidence_refs, ensure_ascii=False, indent=2)

    return f"""You are a security analysis assistant helping to *improve a static analysis*.

Task:
- Diagnose why the *baseline CodeQL analysis* missed (or could not confidently report) the seed vulnerabilities.
- Produce a *falsifiable* hypothesis about the missing "project semantics".

Context:
- Target CWE: {target_cwe_id or 'unknown'}
- We are running a failure-driven loop: baseline -> diagnosis -> semantic patch -> re-run -> measure hit rate.
- Evidence MUST come from the provided structured references and code snippets.
- Do NOT invent CodeQL APIs. If you need CodeQL library docs, you MAY request a doc lookup via an MCP RAG tool,
  but only if necessary. If you did not use docs, return an empty docs_citation_refs.

GapKind taxonomy (choose ONE):
- sink_semantics_missing: the write function / sink isn't modeled as a write or isn't captured.
- capacity_semantics_missing: buffer capacity / destination bound isn't modeled or not comparable.
- length_semantics_missing: length/source bound information isn't modeled (e.g., external string length).
- guard_semantics_missing: safety checks exist but are not modeled; or guards are missing but need modeling.
- logic_mismatch: the query's assumptions don't match the project's idioms; needs alternative modeling.
- unknown: cannot determine from evidence.

Baseline outputs (per seed):
{baseline_digest}

Structured evidence references (paths only; do not quote content unless you read them):
{evidence_digest}

Source snippets around the vulnerable locations:
{chr(10).join(snippet_blocks) if snippet_blocks else '[no snippets available]'}

Output requirements:
- Return ONLY a single JSON object.
- Your hypothesis MUST be testable: it should imply what semantics to add in a .qll patch.
- supporting_evidence_refs should reference some of the given EvidenceRef entries (by repeating them), or add new ones
  if you created additional logs/snippets.

JSON schema:
{json.dumps(_GAP_DIAGNOSIS_SCHEMA_HINT, ensure_ascii=False, indent=2)}
"""


_SEMANTIC_SYNTHESIS_SCHEMA_HINT = {
    "type": "object",
    "properties": {
        "semantic_pack_qll": {"type": "string"},
        "validation_query_ql": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["semantic_pack_qll", "validation_query_ql"],
}


def _build_semantic_synthesis_prompt(
    target_cwe_id: Optional[str],
    diagnosis: Dict[str, Any],
    evidence_refs: List[Dict[str, Any]],
    docs_refs: List[str],
) -> str:
    """Prompt to synthesize a semantic pack (.qll) and a validation query (.ql).

Important: This project is research code; prefer clarity and conservative semantics.
"""
    return f"""You are generating a *project-specific semantic model extension* for CodeQL.

Goal:
- Based on the failure diagnosis, generate:
  1) ProjectSemantics.qll: a semantic extension (library) with predicates/classes that encode the missing semantics.
  2) ValidateOnSeeds.ql: a small validation query that imports ProjectSemantics.qll and produces SARIF results.

Constraints:
- Keep it minimal and readable.
- Do not assume you can change CodeQL standard library files.
- If you rely on CodeQL library APIs, ensure you only use well-known stable imports. If uncertain, keep the query
  self-contained and conservative.

Context:
- Target CWE: {target_cwe_id or 'unknown'}
- Diagnosis:
{json.dumps(diagnosis, ensure_ascii=False, indent=2)}

Evidence references you may use conceptually:
{json.dumps(evidence_refs, ensure_ascii=False, indent=2)}

Doc citations already collected (may be empty):
{json.dumps(docs_refs, ensure_ascii=False, indent=2)}

Failure-driven requirement:
- The generated semantic pack MUST be directly motivated by the diagnosis hypothesis.
- The validation query MUST be able to "observe" the behavior change on seed cases.

Output format:
- Return ONLY a JSON object with fields:
  - semantic_pack_qll (string)
  - validation_query_ql (string)
  - notes (string, optional)

JSON schema:
{json.dumps(_SEMANTIC_SYNTHESIS_SCHEMA_HINT, ensure_ascii=False, indent=2)}
"""


# -----------------------------
# Conservative defaults (used when no LLM key is available)
# -----------------------------


_DEFAULT_BASELINE_QL = r"""/**
 * @name Baseline: calls to strcpy (very coarse)
 * @kind problem
 * @id deepvuln/baseline-strcpy
 */

import cpp

from FunctionCall call
where call.getTarget().getName() = "strcpy"
select call, "Call to strcpy (baseline)."
"""


_DEFAULT_SEMANTIC_PACK_QLL = r"""/**
 * ProjectSemantics.qll (placeholder)
 *
 * This file is intentionally minimal in the skeleton implementation.
 * Replace with your real "semantic pack" modeling hooks.
 */

import cpp

/**
 * Example predicate: treat certain fields as "unconstrained" sources.
 *
 * NOTE: This is a stub. In your real implementation, this predicate should be
 * derived from evidence and optionally backed by doc citations.
 */
predicate isUnconstrainedStringExpr(Expr e) {
  exists(FieldAccess fa | e = fa)
}
"""


_DEFAULT_VALIDATION_QUERY_QL = r"""/**
 * @name Validate seeds: strcpy with likely-unconstrained source
 * @kind problem
 * @id deepvuln/validate-unconstrained-strcpy
 */

import cpp
import ProjectSemantics

from FunctionCall call
where
  call.getTarget().getName() = "strcpy" and
  isUnconstrainedStringExpr(call.getArgument(1))
select call, "strcpy where source is considered unconstrained (semantic pack)."
"""


def _parse_sarif_locations(sarif_path: Path) -> List[Dict[str, Any]]:
    """
    Minimal SARIF parser:
    Extract (uri, startLine, endLine) from results.
    This is intentionally conservative for V1.
    """
    if not sarif_path.exists():
        return []

    try:
        data = json.loads(sarif_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []

    locs: List[Dict[str, Any]] = []
    runs = data.get("runs") or []
    for run in runs:
        results = run.get("results") or []
        for r in results:
            locations = r.get("locations") or []
            for loc in locations:
                phys = (loc.get("physicalLocation") or {})
                artifact = (phys.get("artifactLocation") or {})
                region = (phys.get("region") or {})
                uri = artifact.get("uri")
                start_line = region.get("startLine")
                end_line = region.get("endLine", start_line)
                if uri:
                    locs.append({"uri": uri, "startLine": start_line, "endLine": end_line})
    return locs


def _hit_vuln_site(locs: List[Dict[str, Any]], vuln_file_rel: str, vuln_line: int, radius: int = 8) -> bool:
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