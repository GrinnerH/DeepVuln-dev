from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.agents.llm_client import invoke_llm_json
from app.core.state import DeepVulnState, EvidenceRef, GapDiagnosis, ModelingDecision, ModelingRecord
from app.mcp.codeql_mcp import CodeQLMCPClient, CodeQLExecutionError

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
    """Normalize seed cases.

    Step A (seed provisioning) is automated:
      - clone / checkout vuln commit
      - create CodeQL database
      - locate vulnerable line for prompt snippeting

    This node keeps orchestration logic minimal and delegates robustness to
    `app.utils.seed_bootstrap` + `app.mcp.codeql_mcp`.
    """
    from app.utils.seed_bootstrap import bootstrap_seed_case, load_seed_cases

    workspace_dir = state.get("workspace_dir") or "./artifacts/run"
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    seeds_in_state = state.get("seed_cases") or []
    seeds = seeds_in_state if seeds_in_state else load_seed_cases()
    if not seeds:
        return {"error_log": ["No seed cases found (state.seed_cases empty and seeds/seed_cases.json not found)."]}

    codeql = CodeQLMCPClient()
    enriched: List[Dict[str, Any]] = []
    errors: List[str] = []
    for seed in seeds:
        try:
            enriched_seed = bootstrap_seed_case(seed, workspace_dir=workspace_dir, codeql=codeql, overwrite_db=False)
            enriched.append(_normalize_seed_case(enriched_seed))
        except Exception as exc:
            msg = f"Seed bootstrap failed for {seed.get('cve_id') or seed.get('project_name')}: {exc}"
            logger.exception(msg)
            errors.append(msg)
            enriched.append(_normalize_seed_case(seed))

    update: Dict[str, Any] = {"seed_cases": enriched}
    if errors:
        update["error_log"] = (state.get("error_log") or []) + errors
    return update


# -----------------------------
# Stage B loop: diagnose -> synthesize -> evaluate
# -----------------------------


def diagnose_vulnerability_gap(state: DeepVulnState) -> Dict[str, Any]:
    """Run a baseline query and ask the LLM to produce a failure-driven diagnosis.

    Responsibilities:
    1) Execute baseline analysis using CodeQLMCPClient (official suite or user-provided).
    2) Read the vulnerable code location for each seed (keep State small; embed small snippets).
    3) Construct a *failure-driven* prompt, instructing the LLM to classify a GapKind and
       propose a falsifiable hypothesis.

    Notes on RAG:
    - This node MUST NOT force docs lookups; it should explicitly permit optional doc citation.
    - If/when a RAG MCP tool is wired, the LLM can decide to call it; until then, we leave
      docs_citation_refs empty.
    """
    workspace = Path(state.get("workspace_dir") or "./artifacts/run")
    baseline_dir = workspace / "baseline" / f"iter_{state.get('iteration_count', 0)+1:02d}"
    _ensure_dir(baseline_dir)

    seeds = state.get("seed_cases") or []
    if not seeds:
        logger.warning("No seed_cases provided. Returning unknown diagnosis.")
        return {
            "active_diagnosis": {
                "gap_kind": "unknown",
                "hypothesis": "No seed cases were provided; cannot run baseline diagnosis.",
                "supporting_evidence_refs": [],
                "docs_citation_refs": [],
            }
        }

    codeql = CodeQLMCPClient()

    # --- Step 1: Baseline run (best-effort) ---
    baseline_outputs: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds, start=1):
        seed = _normalize_seed_case(seed)
        db_path = seed.get("codeql_db_path")
        repo_path = seed.get("repo_path")
        vuln_file = seed.get("vuln_file")
        vuln_line = seed.get("vuln_line")

        sarif_out = str(baseline_dir / f"seed_{i:02d}_baseline.sarif")
        baseline_suite = seed.get("baseline_suite") or state.get("baseline_suite")
        baseline_query = seed.get("baseline_query") or state.get("baseline_query")

        baseline_entry: Dict[str, Any] = {
            "seed_index": i,
            "cve_id": seed.get("cve_id"),
            "repo_path": repo_path,
            "db_path": db_path,
            "vuln_file": vuln_file,
            "vuln_line": vuln_line,
            "baseline_suite": baseline_suite,
            "baseline_query": baseline_query,
            "sarif_path": sarif_out,
            "baseline_ok": False,
            "baseline_error": None,
            "baseline_hit": None,
        }

        if not db_path:
            baseline_entry["baseline_error"] = "Missing codeql_db_path for seed."
            baseline_outputs.append(baseline_entry)
            continue

        try:
            if baseline_suite:
                res = codeql.database_analyze(
                    database=db_path,
                    suite_or_query=baseline_suite,
                    sarif_output=sarif_out,
                    additional_args=["--format=sarif-latest"],
                )
            else:
                # Fallback: run a very small built-in baseline query if none specified.
                baseline_query_path = baseline_dir / "baseline_strcpy_calls.ql"
                if not baseline_query_path.exists():
                    baseline_query_path.write_text(
                        _DEFAULT_BASELINE_QL,
                        encoding="utf-8",
                    )
                res = codeql.database_analyze(
                    database=db_path,
                    suite_or_query=str(baseline_query_path),
                    sarif_output=sarif_out,
                    additional_args=["--format=sarif-latest"],
                )
            baseline_entry["baseline_ok"] = True
            baseline_entry["baseline_summary"] = res.get("summary")
        except CodeQLExecutionError as e:
            baseline_entry["baseline_error"] = str(e)

        # Best-effort hit check near vulnerable location.
        if baseline_entry["baseline_ok"] and repo_path and vuln_file and isinstance(vuln_line, int):
            baseline_entry["baseline_hit"] = _sarif_has_hit_near_location(
                sarif_path=sarif_out,
                repo_path=repo_path,
                rel_file=vuln_file,
                vuln_line=vuln_line,
            )

        baseline_outputs.append(baseline_entry)

    # --- Step 2: Prepare evidence snippets (keep small; reference paths for the rest) ---
    evidence_refs: List[EvidenceRef] = []
    snippet_blocks: List[str] = []
    for entry in baseline_outputs:
        repo_path = entry.get("repo_path")
        vuln_file = entry.get("vuln_file")
        vuln_line = entry.get("vuln_line")

        # Always add SARIF path as structured evidence (even if baseline failed).
        if entry.get("sarif_path"):
            evidence_refs.append({
                "kind": "sarif",
                "path": entry["sarif_path"],
                "note": f"Baseline SARIF for seed#{entry.get('seed_index')} ({entry.get('cve_id')})",
            })

        if repo_path and vuln_file and isinstance(vuln_line, int):
            snippet = _read_source_window(repo_path, vuln_file, vuln_line, window=40)
            snippet_blocks.append(
                f"### Seed #{entry.get('seed_index')} {entry.get('cve_id')}\n"
                f"- Vulnerable location: {vuln_file}:{vuln_line}\n"
                f"- Baseline hit near location: {entry.get('baseline_hit')}\n"
                f"- Baseline errors (if any): {entry.get('baseline_error')}\n\n"
                f"```\n{snippet}\n```\n"
            )

    # --- Step 3: Failure-driven prompt ---
    prompt = _build_gap_diagnosis_prompt(
        target_cwe_id=state.get("target_cwe_id"),
        baseline_outputs=baseline_outputs,
        snippet_blocks=snippet_blocks,
        evidence_refs=evidence_refs,
    )

    diagnosis = invoke_llm_json(
        prompt=prompt,
        json_schema_hint=_GAP_DIAGNOSIS_SCHEMA_HINT,
        fallback={
            "gap_kind": "unknown",
            "hypothesis": "LLM call unavailable; please inspect baseline logs and source snippets.",
            "supporting_evidence_refs": evidence_refs,
            "docs_citation_refs": [],
        },
    )

    # Ensure required fields exist.
    active: GapDiagnosis = {
        "gap_kind": diagnosis.get("gap_kind", "unknown"),
        "hypothesis": diagnosis.get("hypothesis", ""),
        "supporting_evidence_refs": diagnosis.get("supporting_evidence_refs", evidence_refs) or evidence_refs,
        "docs_citation_refs": diagnosis.get("docs_citation_refs", []) or [],
    }

    return {"active_diagnosis": active}


def synthesize_semantic_model(state: DeepVulnState) -> Dict[str, Any]:
    """Generate a versioned semantic pack (.qll) based on the active diagnosis.

    This node is intentionally *conservative*:
    - It generates a semantic pack plus a small validation query that uses it.
    - It writes artifacts to workspace_dir/semantic_packs/iter_XX/.

    The content is designed to be swapped later with your "template + semantic pack" split.
    For now it provides a functional, self-contained validation query that can be used by
    evaluate_modeling_success.
    """
    workspace = Path(state.get("workspace_dir") or "./artifacts/run")
    iter_index = int(state.get("iteration_count", 0)) + 1
    out_dir = workspace / "semantic_packs" / f"iter_{iter_index:02d}"
    _ensure_dir(out_dir)

    diagnosis = state.get("active_diagnosis") or {}

    prompt = _build_semantic_synthesis_prompt(
        target_cwe_id=state.get("target_cwe_id"),
        diagnosis=diagnosis,
        evidence_refs=diagnosis.get("supporting_evidence_refs", []),
        docs_refs=diagnosis.get("docs_citation_refs", []),
    )

    synthesis = invoke_llm_json(
        prompt=prompt,
        json_schema_hint=_SEMANTIC_SYNTHESIS_SCHEMA_HINT,
        fallback={
            "semantic_pack_qll": _DEFAULT_SEMANTIC_PACK_QLL,
            "validation_query_ql": _DEFAULT_VALIDATION_QUERY_QL,
            "notes": "LLM unavailable; using conservative default artifacts.",
        },
    )

    qll_path = out_dir / "ProjectSemantics.qll"
    ql_path = out_dir / "ValidateOnSeeds.ql"
    qll_path.write_text(str(synthesis.get("semantic_pack_qll", "")), encoding="utf-8")
    ql_path.write_text(str(synthesis.get("validation_query_ql", "")), encoding="utf-8")

    # Keep compile logs separate.
    compile_log_path = out_dir / "compile.log"

    active_artifact = {
        "semantic_pack_path": str(qll_path),
        "compile_log_ref": {"kind": "diagnostic_log", "path": str(compile_log_path), "note": "CodeQL compilation log"},
        # non-schema but useful for evaluation
        "validation_query_path": str(ql_path),
    }

    return {"active_semantic_artifact": active_artifact}


def evaluate_modeling_success(state: DeepVulnState) -> Dict[str, Any]:
    """Compile and run the generated patch against seeds, then decide converge/continue/halt.

    Evaluation policy (V1):
    - Compile must succeed (compile_ok).
    - seed_hit_rate is computed as (#seeds with hit near vuln location) / (#seeds with known location).
    - Converged if compile_ok and seed_hit_rate == 1.0.
    - Continue if compile_ok and seed_hit_rate improves or iterations remain.
    - Halt if compile fails repeatedly or iterations exhausted.
    """
    workspace = Path(state.get("workspace_dir") or "./artifacts/run")
    iter_index = int(state.get("iteration_count", 0)) + 1
    out_dir = workspace / "semantic_packs" / f"iter_{iter_index:02d}"
    _ensure_dir(out_dir)

    seeds = state.get("seed_cases") or []
    artifact = state.get("active_semantic_artifact") or {}
    validation_query = artifact.get("validation_query_path") or ""
    semantic_pack = artifact.get("semantic_pack_path") or ""

    codeql = CodeQLMCPClient()

    # --- Compile step (best-effort) ---
    compile_log_path = out_dir / "compile.log"
    compile_ok = False
    try:
        compile_res = codeql.query_compile(query=validation_query, additional_args=[])
        compile_log_path.write_text(compile_res.get("stdout", "") + "\n" + compile_res.get("stderr", ""), encoding="utf-8")
        compile_ok = True
    except Exception as e:
        compile_log_path.write_text(str(e), encoding="utf-8")
        compile_ok = False

    # --- Run query on each seed and compute hit rate ---
    hits = 0
    denom = 0
    per_seed: List[Dict[str, Any]] = []
    analyze_dir = workspace / "reports" / f"iter_{iter_index:02d}"
    _ensure_dir(analyze_dir)

    for i, seed in enumerate(seeds, start=1):
        seed = _normalize_seed_case(seed)
        db_path = seed.get("codeql_db_path")
        repo_path = seed.get("repo_path")
        vuln_file = seed.get("vuln_file")
        vuln_line = seed.get("vuln_line")

        sarif_out = str(analyze_dir / f"seed_{i:02d}_validation.sarif")
        entry = {"seed_index": i, "cve_id": seed.get("cve_id"), "sarif_path": sarif_out, "ok": False, "hit": False}

        if not db_path or not validation_query:
            entry["error"] = "Missing database or validation query."
            per_seed.append(entry)
            continue

        try:
            codeql.database_analyze(
                database=db_path,
                suite_or_query=validation_query,
                sarif_output=sarif_out,
                additional_args=["--format=sarif-latest"],
            )
            entry["ok"] = True
        except CodeQLExecutionError as e:
            entry["error"] = str(e)
            entry["ok"] = False

        if repo_path and vuln_file and isinstance(vuln_line, int):
            denom += 1
            entry["hit"] = _sarif_has_hit_near_location(sarif_out, repo_path, vuln_file, vuln_line)
            if entry["hit"]:
                hits += 1

        per_seed.append(entry)

    seed_hit_rate = float(hits) / float(denom) if denom else 0.0

    # --- Decision policy ---
    max_iters = int(state.get("max_iterations", 3))
    outcome: ModelingDecision
    rationale = []

    if not compile_ok:
        outcome = {"outcome": "continue" if iter_index < max_iters else "halt", "rationale": "Compilation failed."}
        rationale.append(f"compile_ok=false (see {compile_log_path})")
    elif seed_hit_rate >= 1.0 and denom > 0:
        outcome = {"outcome": "converged", "rationale": "All seeds were hit near their vulnerable locations."}
        rationale.append(f"seed_hit_rate=1.0 over {denom} seeds")
    else:
        # continue if we have budget; otherwise halt.
        if iter_index < max_iters:
            outcome = {"outcome": "continue", "rationale": "Not all seeds hit; refining semantics."}
        else:
            outcome = {"outcome": "halt", "rationale": "Iteration budget exhausted without convergence."}
        rationale.append(f"seed_hit_rate={seed_hit_rate:.3f} ({hits}/{denom})")

    # --- Append modeling record (for paper artifacts) ---
    record: ModelingRecord = {
        "iteration_index": iter_index,
        "diagnosis": state.get("active_diagnosis", {}),
        "semantic_artifact": {
            "semantic_pack_path": semantic_pack,
            "compile_log_ref": {"kind": "diagnostic_log", "path": str(compile_log_path), "note": "Compilation log"},
        },
        "metrics": {
            "compile_ok": compile_ok,
            "seed_hit_rate": seed_hit_rate,
        },
        "decision": outcome,
    }

    return {
        "modeling_history": [record],
        "iteration_count": iter_index,
        "is_converged": True if outcome.get("outcome") == "converged" else False,
        "active_decision": outcome,
        "best_semantic_pack_path": semantic_pack if outcome.get("outcome") == "converged" else state.get("best_semantic_pack_path", ""),
        # store a compact evaluation summary for debugging
        "scanning_result": state.get("scanning_result", {}),
    }


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
