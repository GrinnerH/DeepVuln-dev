from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from jsonschema import Draft202012Validator

from app.agent.model import build_chat_model, build_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.tools.file_management import ReadFileTool, ListDirectoryTool
from langchain_community.tools.shell import ShellTool
from app.core.state import DeepVulnState, EvidenceRef, GapDiagnosis, ModelingDecision, ModelingRecord
from app.mcp.codeql_mcp import CodeQLMCPClient, CodeQLExecutionError, codeql_analyze_tool
from app.mcp.ragflow_mcp import ragflow_search_tool
from utils.io import ensure_dir, safe_read_text
from utils.jsonx import ensure_list, json_loads_best_effort
from app.semantics.builder import build_semantic_pack
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
        content = safe_read_text(file_path)
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


def _strip_markdown_fences(text: str) -> str:
    if "```" not in text:
        return text
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].lstrip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _contains_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def _maybe_localize_query(path_str: str, *, workspace_dir: Path) -> str:
    if not path_str:
        return path_str
    if not _contains_non_ascii(path_str):
        return path_str
    src = Path(path_str)
    if not src.exists():
        return path_str
    ws = workspace_dir.resolve()
    dest_dir = ensure_dir(ws / "baseline")
    dest = dest_dir / _sanitize_filename(src.name)
    try:
        shutil.copy2(src, dest)
        return str(dest.resolve())
    except Exception:
        return path_str


# -----------------------------
# Stage A: setup + ingest
# -----------------------------


def setup_environment(state: DeepVulnState) -> Dict[str, Any]:
    """Create run workspace directories and initialize control fields.

    Notes:
      - This node returns only minimal, serializable fields for state updates.
      - Runtime logs are written to a local trace file (not stored in state).
    """
    workspace = Path(state.get("workspace_dir") or "./artifacts/run").resolve()
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
    Path(workspace_dir).resolve().mkdir(parents=True, exist_ok=True)

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
    workspace_dir = Path(state["workspace_dir"]).resolve()
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
    repo_path = Path(seed0["repo_path"]).expanduser().resolve()
    db_path = str(Path(seed0["db_path"]).expanduser().resolve())
    vuln_file_rel = seed0["vulnerable_file"]
    vuln_line = int(seed0.get("vulnerable_line") or 1)

    baseline_query_path = seed0.get("baseline_query_path") or os.getenv("BASELINE_QUERY_PATH", "")
    if baseline_query_path:
        baseline_query_path = str(Path(baseline_query_path).expanduser().resolve())
        baseline_query_path = _maybe_localize_query(
            baseline_query_path, workspace_dir=workspace_dir
        )
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
    logger.info(
        "[diagnose] vuln_file_rel=%s abs_vuln_file=%s exists=%s",
        vuln_file_rel,
        abs_vuln_file,
        abs_vuln_file.exists(),
    )
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
    baseline_query_body = ""
    if baseline_query_path:
        try:
            baseline_query_body = safe_read_text(Path(baseline_query_path))
        except Exception:
            baseline_query_body = ""
    baseline_sarif_summary = _summarize_sarif_results(baseline_sarif_path)

    prompt = _DIAGNOSE_PROMPT_TEMPLATE.format(
        cve_id=cve_id,
        baseline_summary=baseline_summary,
        baseline_log_path=str(baseline_log_path),
        baseline_sarif_path=str(baseline_sarif_path),
        baseline_sarif_summary=baseline_sarif_summary,
        baseline_query_body=baseline_query_body,
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
    print(f"[STEP1] raw_text:\n{raw_text}")

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

    # ---- 4) Pattern abstraction agent (skills + tools) ----
    class _AgentToolLogger(BaseCallbackHandler):
        def __init__(self) -> None:
            self.records: List[Dict[str, Any]] = []

        def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
            self.records.append(
                {"event": "tool_start", "name": serialized.get("name"), "input": input_str}
            )

        def on_tool_end(self, output: str, **kwargs: Any) -> None:
            self.records.append({"event": "tool_end", "output": output})

    repo_root = Path(".").resolve()
    fs_tools = [
        ReadFileTool(root_dir=str(repo_root)),
        ListDirectoryTool(root_dir=str(repo_root)),
    ]
    shell_tool = ShellTool()
    tools = [load_skill, ragflow_search_tool, codeql_analyze_tool, *fs_tools, shell_tool]

    agent_system_prompt = (
        "You are a diagnosis-and-abstraction agent.\n"
        "Always call load_skill('diagnose_and_abstract') first and follow it strictly.\n"
        "You MUST call ragflow_search_tool to align with CodeQL docs before final output.\n"
        "You may use file tools and ShellTool for read-only commands (ls, rg, find, cat, sed).\n"
        "Do NOT modify files or run destructive commands.\n"
        "Return ONLY JSON as specified by the skill.\n"
    )
    agent = create_agent(
        model=build_model(),
        tools=tools,
        system_prompt=agent_system_prompt,
    )

    agent_input = "\n".join(
        [
            "Seed diagnosis context:",
            json.dumps(diagnosis, ensure_ascii=False, indent=2),
            "",
            f"Repo path: {repo_path}",
            f"Vulnerable file: {vuln_file_rel}",
            f"Vulnerable line: {vuln_line}",
            f"Baseline query path: {baseline_query_path}",
            f"Baseline SARIF path: {baseline_sarif_path}",
            "",
            "Code window:",
            code_snip,
        ]
    )

    tool_logger = _AgentToolLogger()
    try:
        agent_result = agent.invoke({"input": agent_input}, config={"callbacks": [tool_logger]})
        agent_text = agent_result.get("output") if isinstance(agent_result, dict) else str(agent_result)
    except Exception as exc:
        agent_text = f"<<AGENT_FAILURE error={exc}>>"

    parsed_patterns = json_loads_best_effort(_strip_markdown_fences(agent_text).strip(), fallback={})
    learned_patterns = parsed_patterns.get("pattern_hypotheses") if isinstance(parsed_patterns, dict) else None
    if not isinstance(learned_patterns, list):
        learned_patterns = []

    pattern_path = diagnostics_dir / f"{cve_id}_pattern_hypotheses.json"
    pattern_payload = {
        "raw_output": agent_text,
        "parsed": parsed_patterns,
        "tool_log": tool_logger.records,
    }
    pattern_path.write_text(json.dumps(pattern_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnosis["pattern_hypotheses_ref"] = str(pattern_path)
    diagnosis["learned_patterns"] = learned_patterns
    logger.info("[diagnose] wrote pattern hypotheses: %s", pattern_path)

    return {"active_diagnosis": diagnosis}


def synthesize_semantic_model(state: DeepVulnState) -> Dict[str, Any]:
    workspace_dir = Path(state["workspace_dir"]).resolve()
    ensure_dir(workspace_dir)

    seed_cases = state.get("seed_cases", [])
    if not seed_cases:
        return {"active_semantic_artifact": {"semantic_pack_path": ""}}

    seed0 = seed_cases[0]
    cve_id = seed0.get("cve_id") or "UNKNOWN_CVE"

    diagnosis = state.get("active_diagnosis") or {}
    iter_idx = int(state.get("iteration_count", 0)) + 1
    pack_dir = ensure_dir(workspace_dir / "semantic_packs" / f"iter_{iter_idx:02d}" / "pack")
    diagnostics_dir = ensure_dir(workspace_dir / "diagnostics")

    baseline_query_path = seed0.get("baseline_query_path") or os.getenv("BASELINE_QUERY_PATH", "")
    if baseline_query_path:
        baseline_query_path = str(Path(baseline_query_path).expanduser().resolve())
        baseline_query_path = _maybe_localize_query(baseline_query_path, workspace_dir=workspace_dir)

    candidates_path = diagnostics_dir / f"{cve_id}_candidates_iter_{iter_idx:02d}.json"
    if candidates_path.exists():
        candidates = json_loads_best_effort(candidates_path.read_text(encoding="utf-8"), fallback={})
    else:
        candidates = {
            "wrapper_candidates": [],
            "capacity_candidates": [],
            "guard_candidates": [],
        }
        candidates_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    prompt_path = Path("app/semantics/prompts/annotate_candidates.md")
    prompt_body = safe_read_text(prompt_path) if prompt_path.exists() else _ANNOTATE_PROMPT_FALLBACK
    prompt = "\n\n".join(
        [
            prompt_body.strip(),
            "",
            "Learned patterns (prioritize these over raw diagnosis):",
            json.dumps(learned_patterns, ensure_ascii=False, indent=2),
            "",
            "SemanticCandidates JSON:",
            json.dumps(candidates, ensure_ascii=False, indent=2),
        ]
    )

    runnable = build_chat_model(structured_schema=None)
    messages = [
        SystemMessage(content="You output SemanticFacts JSON only."),
        HumanMessage(content=prompt),
    ]
    raw = runnable.invoke(messages)
    raw_text = getattr(raw, "content", raw)
    raw_str = str(raw_text)
    sanitized = _strip_markdown_fences(raw_str).strip()
    semantic_facts = json_loads_best_effort(sanitized, fallback={})
    if not isinstance(semantic_facts, dict):
        semantic_facts = {}

    facts_schema_path = Path("app/semantics/schemas/semantic_facts.schema.json")
    facts_schema = json_loads_best_effort(
        safe_read_text(facts_schema_path),
        fallback={},
    )
    schema_errors: List[str] = []
    if not facts_schema:
        schema_errors.append("SemanticFacts schema missing or unreadable.")
    else:
        validator = Draft202012Validator(facts_schema)
        schema_errors.extend([e.message for e in sorted(validator.iter_errors(semantic_facts), key=lambda e: e.path)])

    if schema_errors:
        invalid_path = diagnostics_dir / f"{cve_id}_facts_invalid_iter_{iter_idx:02d}.json"
        invalid_payload = {
            "raw_content": raw_str,
            "parsed": semantic_facts,
            "errors": schema_errors,
        }
        invalid_path.write_text(json.dumps(invalid_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[synthesize] semantic_facts invalid; wrote %s", invalid_path)
        return {
            "active_semantic_artifact": {
                "semantic_pack_path": "",
                "facts_valid": False,
                "facts_invalid_path": str(invalid_path),
            },
            "active_decision": {
                "outcome": "continue",
                "rationale": "SemanticFacts failed schema validation; see facts_invalid path.",
            },
        }

    semantic_facts.setdefault("wrapper_models", [])
    semantic_facts.setdefault("capacity_macros", [])
    semantic_facts.setdefault("capacity_fields", [])
    semantic_facts.setdefault("guard_patterns", [])

    if not baseline_query_path:
        raise ValueError("baseline_query_path is required to build semantic pack.")

    build_result = build_semantic_pack(
        pack_dir=Path(pack_dir),
        baseline_query_path=Path(baseline_query_path),
        semantic_facts=semantic_facts,
    )

    logger.info("[synthesize] wrote semantic pack: %s", build_result.get("pack_dir"))

    artifact = {
        "semantic_pack_path": build_result.get("pack_dir", ""),
    }
    return {"active_semantic_artifact": artifact}


def evaluate_modeling_success(state: DeepVulnState) -> Dict[str, Any]:
    workspace_dir = Path(state["workspace_dir"]).resolve()
    ensure_dir(workspace_dir)

    seed_cases = state.get("seed_cases", [])
    if not seed_cases:
        decision = {"outcome": "halt", "rationale": "No seed_cases; cannot evaluate modeling success."}
        return {"active_decision": decision, "is_converged": False}

    seed0 = seed_cases[0]
    cve_id = seed0.get("cve_id") or "UNKNOWN_CVE"

    # --- Canonical schema only (NO key-guessing) ---
    db_path = str(Path(seed0["db_path"]).expanduser().resolve())
    vuln_file_rel = seed0["vulnerable_file"]
    vuln_line = int(seed0.get("vulnerable_line") or 1)
    codeql_search_paths = seed0.get("codeql_search_paths") or []

    iter_idx = int(state.get("iteration_count", 0)) + 1
    artifact = state.get("active_semantic_artifact") or {}
    if artifact.get("facts_valid") is False:
        decision: ModelingDecision = {
            "outcome": "continue",
            "rationale": "SemanticFacts invalid; refine and retry.",
        }
        record: ModelingRecord = {
            "iteration_index": iter_idx,
            "diagnosis": state.get("active_diagnosis") or {},
            "semantic_artifact": artifact,
            "metrics": {"compile_ok": False, "seed_hit_rate": 0.0},
            "decision": decision,
        }
        return {
            "active_decision": decision,
            "modeling_history": [record],
            "iteration_count": iter_idx,
            "is_converged": False,
        }

    pack_dir = artifact.get("semantic_pack_path")
    if not pack_dir:
        decision = {"outcome": "halt", "rationale": "No active semantic pack path to compile/analyze."}
        return {"active_decision": decision, "is_converged": False}

    pack_dir_p = Path(pack_dir)
    diagnostics_dir = ensure_dir(workspace_dir / "diagnostics")
    compile_log_path = diagnostics_dir / f"{cve_id}_compile_iter_{iter_idx:02d}.json"
    probe_sarif_path = diagnostics_dir / f"{cve_id}_compile_iter_{iter_idx:02d}.sarif"
    overlay_sarif_path = diagnostics_dir / f"{cve_id}_overlay_iter_{iter_idx:02d}.sarif"
    overlay_log_path = diagnostics_dir / f"{cve_id}_overlay_iter_{iter_idx:02d}.json"
    regression_sarif_path = diagnostics_dir / f"{cve_id}_regression_iter_{iter_idx:02d}.sarif"
    regression_log_path = diagnostics_dir / f"{cve_id}_regression_iter_{iter_idx:02d}.json"

    # Local client (do NOT store in state; keep state serializable)
    codeql = CodeQLMCPClient()

    # ---- 1) Compile the generated semantic pack via a probe query ----
    probe_path = pack_dir_p / "_compile_probe.ql"
    probe_src = "\n".join(
        [
            "/**",
            " * @kind diagnostic",
            " * @id deepvuln/compile-probe",
            " */",
            "import cpp",
            "import ProjectSemantics",
            "",
            "from Function f",
            "select f, f.getName()",
            "",
        ]
    )
    if not probe_path.exists() or probe_path.read_text(encoding="utf-8", errors="ignore") != probe_src:
        probe_path.write_text(probe_src, encoding="utf-8")

    search_paths = list(codeql_search_paths)
    if str(pack_dir_p) not in search_paths:
        search_paths.append(str(pack_dir_p))

    compile_res = codeql.analyze_database(
        db_path=str(db_path),
        query_or_suite_path=str(probe_path),
        output_sarif_path=str(probe_sarif_path),
        additional_search_paths=search_paths or None,
        extra_args=["--rerun"],
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
    overlay_query_path = pack_dir_p / "queries" / "_regression_semantic_overlay.ql"
    overlay_res: Dict[str, Any]
    if overlay_query_path.exists():
        overlay_res = codeql.analyze_database(
            db_path=str(db_path),
            query_or_suite_path=str(overlay_query_path),
            output_sarif_path=str(overlay_sarif_path),
            additional_search_paths=search_paths or None,
            extra_args=["--rerun"],
            cwd=None,
        )
    else:
        overlay_res = {
            "ok": False,
            "returncode": 1,
            "stderr": "Missing pack overlay query; cannot run overlay check.",
            "stdout": "",
            "cmd": [],
        }
    overlay_res["semantic_pack_path"] = str(pack_dir_p)
    overlay_res["overlay_query_path"] = str(overlay_query_path)
    logger.info("[evaluate] overlay_ok=%s sarif=%s", overlay_res.get("ok"), overlay_sarif_path)

    regression_query_path = pack_dir_p / "queries" / "_regression_with_semantics.ql"
    regression_res: Dict[str, Any]
    if regression_query_path.exists():
        regression_res = codeql.analyze_database(
            db_path=str(db_path),
            query_or_suite_path=str(regression_query_path),
            output_sarif_path=str(regression_sarif_path),
            additional_search_paths=search_paths or None,
            extra_args=["--rerun"],
            cwd=None,
        )
    else:
        regression_res = {
            "ok": False,
            "returncode": 1,
            "stderr": "Missing pack regression query; cannot run regression.",
            "stdout": "",
            "cmd": [],
        }
    regression_res["semantic_pack_path"] = str(pack_dir_p)
    regression_res["regression_query_path"] = str(regression_query_path)
    regression_log_path.write_text(json.dumps(regression_res, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[evaluate] regression_ok=%s sarif=%s", regression_res.get("ok"), regression_sarif_path)

    # ---- 3) Compute seed hit rate (proxy oracle) ----
    overlay_count = 0
    overlay_notifications = 0
    if overlay_sarif_path.exists():
        overlay_count = _count_sarif_results(overlay_sarif_path)
        overlay_notifications = _count_sarif_tool_notifications(overlay_sarif_path)
        overlay_res["overlay_results_count"] = overlay_count
        overlay_res["overlay_notifications_count"] = overlay_notifications
        overlay_log_path.write_text(json.dumps(overlay_res, ensure_ascii=False, indent=2), encoding="utf-8")

    reg_locs = _parse_sarif_locations(regression_sarif_path) if regression_sarif_path.exists() else []
    hit = _hit_vuln_site(reg_locs, vuln_file_rel=vuln_file_rel, vuln_line=vuln_line, radius=3)
    seed_hit_rate = 1.0 if hit else 0.0

    outcome: ModelingDecision["outcome"]
    if hit:
        outcome = "converged"
        rationale = "Regression run produced a SARIF location near the vulnerable site."
        best_pack = str(pack_dir_p)
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
        "metrics": {"compile_ok": True, "seed_hit_rate": seed_hit_rate, "overlay_modeled_count": overlay_count},
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
# Stage C: target scanning (placeholder for MVP)
# -----------------------------


def scan_target_repository(state: DeepVulnState) -> Dict[str, Any]:
    """
    Scan the target repository with the converged semantic pack.

    MVP behavior:
      - If target_repo_path or semantic pack is missing, mark as skipped.
      - This keeps the graph executable without enforcing Step C yet.
    """
    workspace_dir = Path(state.get("workspace_dir") or "./artifacts/run").resolve()
    ensure_dir(workspace_dir)

    target_repo_path = state.get("target_repo_path") or ""
    semantic_pack = state.get("best_semantic_pack_path") or ""

    scanning_result: Dict[str, Any] = {
        "status": "skipped",
        "reason": "",
        "target_repo_path": target_repo_path,
        "semantic_pack_path": semantic_pack,
    }

    if not semantic_pack:
        scanning_result["reason"] = "No best_semantic_pack_path; skipping target scan."
        return {"scanning_result": scanning_result}

    if not target_repo_path:
        scanning_result["reason"] = "No target_repo_path provided; skipping target scan."
        return {"scanning_result": scanning_result}

    # TODO: Integrate Step C database creation + analyze flow.
    scanning_result["reason"] = "Step C scan not implemented in MVP."
    return {"scanning_result": scanning_result}


def finalize_and_report(state: DeepVulnState) -> Dict[str, Any]:
    """
    Write a minimal run summary to disk for auditability.
    """
    workspace_dir = Path(state.get("workspace_dir") or "./artifacts/run").resolve()
    ensure_dir(workspace_dir)
    reports_dir = ensure_dir(workspace_dir / "reports")

    summary = {
        "run_id": state.get("run_id"),
        "iteration_count": state.get("iteration_count", 0),
        "is_converged": state.get("is_converged", False),
        "best_semantic_pack_path": state.get("best_semantic_pack_path", ""),
        "active_decision": state.get("active_decision", {}),
        "modeling_history_len": len(state.get("modeling_history", []) or []),
        "scanning_result": state.get("scanning_result", {}),
    }

    report_path = reports_dir / "run_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[finalize] wrote summary report: %s", report_path)

    scanning_result = dict(state.get("scanning_result", {}) or {})
    scanning_result["report_path"] = str(report_path)
    return {"scanning_result": scanning_result}


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


def _count_sarif_results(sarif_path: Path) -> int:
    """Count SARIF results across all runs."""
    try:
        sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    total = 0
    runs = sarif.get("runs") or []
    for run in runs:
        total += len(run.get("results") or [])
    return total


def _count_sarif_tool_notifications(sarif_path: Path) -> int:
    """Count toolExecutionNotifications across all runs."""
    try:
        sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    total = 0
    runs = sarif.get("runs") or []
    for run in runs:
        invocations = run.get("invocations") or []
        for inv in invocations:
            total += len(inv.get("toolExecutionNotifications") or [])
    return total


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
- Baseline SARIF summary: {baseline_sarif_summary}
- Baseline query body:
{baseline_query_body}

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


def _summarize_sarif_results(sarif_path: Path) -> str:
    try:
        sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"<<FAILED_TO_READ_SARIF error={exc}>>"
    runs = sarif.get("runs") or []
    total = 0
    rule_ids: Dict[str, int] = {}
    for run in runs:
        results = run.get("results") or []
        total += len(results)
        for r in results:
            rid = r.get("ruleId") or "UNKNOWN_RULE"
            rule_ids[rid] = rule_ids.get(rid, 0) + 1
    return json.dumps({"total_results": total, "rule_counts": rule_ids}, ensure_ascii=False)


_ANNOTATE_PROMPT_FALLBACK = (
    "You are a semantic adapter annotator.\n\n"
    "Your input is SemanticCandidates JSON (wrapper/capacity/guard candidates).\n"
    "Your task is to output SemanticFacts JSON only, matching the schema.\n\n"
    "Rules:\n"
    "- Output JSON only (no markdown).\n"
    "- Do not output any CodeQL code.\n"
    "- The output must include wrapper_models, capacity_macros, capacity_fields, guard_patterns.\n"
    "- If unsure, output empty arrays for wrapper_models/capacity_macros/capacity_fields/guard_patterns.\n"
    "\n"
    "Example:\n"
    "{\n"
    "  \"wrapper_models\": [],\n"
    "  \"capacity_macros\": [],\n"
    "  \"capacity_fields\": [],\n"
    "  \"guard_patterns\": []\n"
    "}\n"
)
    learned_patterns = diagnosis.get("learned_patterns") or []
