# app/utils/seed_loader.py
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class SeedMaterialError(RuntimeError):
    """Raised when a seed case is missing required material or cannot be prepared."""
    pass


@dataclass(frozen=True)
class SeedPaths:
    seeds_json: Path
    repos_dir: Path
    codeql_db_dir: Path
    queries_dir: Path


def default_seed_paths(workspace_dir: str) -> SeedPaths:
    ws = Path(workspace_dir)
    return SeedPaths(
        seeds_json=Path("seeds/seed_cases.json"),
        repos_dir=ws / "repos",
        codeql_db_dir=ws / "codeql_dbs",
        queries_dir=ws / "baseline",
    )


# -----------------------------
# JSON helpers
# -----------------------------

def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SeedMaterialError(f"Seed JSON does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SeedMaterialError(f"Seed JSON must be a list of objects. Got: {type(data)}")
    return data  # type: ignore[return-value]


def load_seed_cases(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """Load seed cases from a JSON list file."""
    return _load_json_array(Path(path))


def save_seed_cases(path: str | os.PathLike[str], seeds: List[Dict[str, Any]]) -> None:
    """Save seed cases as a JSON list file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(seeds, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# Normalization (canonical schema)
# -----------------------------

def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _resolve_repo_path(paths: SeedPaths, seed: Dict[str, Any]) -> Path:
    # Priority: explicit repo_path -> derive from project_name
    if seed.get("repo_path"):
        return Path(seed["repo_path"]).expanduser()
    project = seed.get("project_name") or seed.get("cve_id") or "unknown"
    return paths.repos_dir / _slugify(str(project))


def _resolve_db_path(paths: SeedPaths, seed: Dict[str, Any]) -> Path:
    if seed.get("db_path"):
        return Path(seed["db_path"]).expanduser()
    project = seed.get("project_name") or seed.get("cve_id") or "unknown"
    return paths.codeql_db_dir / _slugify(str(project))


def _resolve_baseline_query(paths: SeedPaths, seed: Dict[str, Any]) -> Optional[Path]:
    # Highest priority: explicit in seed JSON
    explicit = seed.get("baseline_query_path") or seed.get("baseline_query") or seed.get("baseline_suite")
    if explicit:
        return Path(explicit).expanduser()

    # Next: environment variable
    env_path = os.environ.get("BASELINE_QUERY_PATH")
    if env_path:
        return Path(env_path).expanduser()

    # Fallback: a local baseline/ directory under workspace
    if paths.queries_dir.exists():
        return paths.queries_dir
    return None


def normalize_seed_case(workspace_dir: str, seed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a canonical seed dict to avoid key-guessing downstream.
    This is the ONLY schema downstream nodes should read.
    """
    paths = default_seed_paths(workspace_dir)

    required = ["project_name", "cve_id", "cwe_id", "repo_url", "vulnerable_file", "target_function"]
    missing = [k for k in required if not seed.get(k)]
    if missing:
        raise SeedMaterialError(f"Seed is missing required fields: {missing}. seed={seed}")

    repo_path = _resolve_repo_path(paths, seed)
    db_path = _resolve_db_path(paths, seed)
    baseline_query = _resolve_baseline_query(paths, seed)

    # keep extras for forward-compat, but downstream should NOT depend on them
    extra_keys = {
        "project_name", "cve_id", "cwe_id", "repo_url", "repo_path", "db_path",
        "vulnerable_file", "target_function",
        "vulnerable_line", "vuln_line", "line",
        "vuln_commit_sha", "vulnerable_commit_sha", "commit_sha",
        "fix_commit_sha", "patch_commit_sha",
        "baseline_query_path", "baseline_query", "baseline_suite",
        "regression_query_path", "codeql_search_paths",
    }

    normalized: Dict[str, Any] = {
        # identity
        "project_name": seed["project_name"],
        "cve_id": seed["cve_id"],
        "cwe_id": seed["cwe_id"],
        "repo_url": seed["repo_url"],

        # repo/db material
        "repo_path": str(repo_path),
        "db_path": str(db_path),

        # vuln anchor material
        "vulnerable_file": seed["vulnerable_file"],
        "target_function": seed["target_function"],
        "vulnerable_line": seed.get("vulnerable_line") or seed.get("vuln_line") or seed.get("line"),

        # commit pins (optional but recommended)
        "vuln_commit_sha": seed.get("vuln_commit_sha") or seed.get("vulnerable_commit_sha") or seed.get("commit_sha"),
        "fix_commit_sha": seed.get("fix_commit_sha") or seed.get("patch_commit_sha"),

        # analysis entry points
        "baseline_query_path": str(baseline_query) if baseline_query else None,
        # V1: regression defaults to baseline
        "regression_query_path": str(baseline_query) if baseline_query else None,

        # optional CodeQL search paths (project-controlled)
        "codeql_search_paths": seed.get("codeql_search_paths", []),

        # preserve everything else for later use/debug
        "_extra": {k: v for k, v in seed.items() if k not in extra_keys},
    }
    return normalized


def load_and_normalize_seeds(workspace_dir: str) -> List[Dict[str, Any]]:
    paths = default_seed_paths(workspace_dir)
    raw = _load_json_array(paths.seeds_json)
    return [normalize_seed_case(workspace_dir, s) for s in raw]


# -----------------------------
# Step A (bootstrap) helpers
# -----------------------------

def _run(cmd: List[str], *, cwd: Optional[str] = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _ensure_git_checkout(repo_url: str, repo_path: Path, commit_sha: Optional[str]) -> None:
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    if not repo_path.exists():
        _run(["git", "clone", "--recursive", repo_url, str(repo_path)])

    # best-effort fetch (non-fatal)
    try:
        _run(["git", "fetch", "--all", "--tags"], cwd=str(repo_path))
    except Exception:
        pass

    if commit_sha:
        _run(["git", "checkout", commit_sha], cwd=str(repo_path))
        # best-effort submodule update
        try:
            _run(["git", "submodule", "update", "--init", "--recursive"], cwd=str(repo_path))
        except Exception:
            pass


def _best_effort_find_vuln_line(repo_path: Path, vulnerable_file: str, target_function: str) -> Optional[int]:
    f = repo_path / vulnerable_file
    if not f.exists():
        f2 = Path(vulnerable_file)
        if f2.exists():
            f = f2
        else:
            return None

    try:
        lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    # Heuristic 1: first occurrence of target_function string
    for i, line in enumerate(lines, start=1):
        if target_function in line:
            return i

    # Heuristic 2: common sink calls (fallback)
    sinks = ["strcpy", "strncpy", "memcpy", "memmove", "sprintf", "snprintf", "strcat", "strncat"]
    for i, line in enumerate(lines, start=1):
        if any(s in line for s in sinks):
            return i
    return None


def bootstrap_seed_case(
    *,
    workspace_dir: str,
    seed: Dict[str, Any],
    codeql: Any,
    overwrite_db: bool = False,
) -> Dict[str, Any]:
    """
    Prepare repo checkout + CodeQL DB + best-effort vuln line, and return canonical seed.
    `codeql` is expected to be CodeQLMCPClient (or a compatible adapter).
    """
    normalized = normalize_seed_case(workspace_dir, seed)

    repo_path = Path(normalized["repo_path"])
    db_path = Path(normalized["db_path"])
    repo_url = normalized["repo_url"]
    commit_sha = normalized.get("vuln_commit_sha")

    # 1) repo checkout
    _ensure_git_checkout(repo_url, repo_path, commit_sha)

    # 2) CodeQL database
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite_db and db_path.exists():
        import shutil
        shutil.rmtree(db_path, ignore_errors=True)

    if not db_path.exists():
        build_cmd = seed.get("build_command") or normalized.get("_extra", {}).get("build_command")
        codeql.create_database(
            source_root=str(repo_path),
            db_path=str(db_path),
            language="cpp",
            command=build_cmd,
            overwrite=False,
            cwd=str(repo_path),
        )

    # 3) vuln line heuristic (for prompt windowing)
    if not normalized.get("vulnerable_line"):
        line = _best_effort_find_vuln_line(
            repo_path,
            normalized["vulnerable_file"],
            normalized["target_function"],
        )
        if line:
            normalized["vulnerable_line"] = int(line)

    return normalized
