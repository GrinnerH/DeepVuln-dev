# app/utils/seed_loader.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class SeedMaterialError(RuntimeError):
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
        seeds_json=Path("seeds") / "seed_cases.json",
        repos_dir=ws / "repos",
        codeql_db_dir=ws / "codeql_dbs",
        queries_dir=Path("queries"),  # project-root relative
    )


def _load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SeedMaterialError(
            f"seed_cases.json not found at: {path}. "
            f"Expected a JSON array of seed cases."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SeedMaterialError(f"Failed to parse JSON: {path}") from e

    if not isinstance(data, list):
        raise SeedMaterialError(f"Expected a JSON array in {path}, got: {type(data)}")
    return data


def _slugify(name: str) -> str:
    keep = []
    for ch in name.strip():
        if ch.isalnum():
            keep.append(ch.lower())
        elif ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    s = "".join(keep).strip("_")
    return s or "repo"


def _resolve_repo_path(paths: SeedPaths, seed: Dict[str, Any]) -> Path:
    # 1) explicit repo_path in seed json
    if seed.get("repo_path"):
        return Path(seed["repo_path"]).expanduser().resolve()

    # 2) infer from workspace repos_dir + project_name
    project = seed.get("project_name") or seed.get("repo_url") or "repo"
    repo_dir = paths.repos_dir / _slugify(str(project))
    return repo_dir.resolve()


def _resolve_db_path(paths: SeedPaths, seed: Dict[str, Any]) -> Path:
    # 1) explicit db_path in seed json
    if seed.get("db_path"):
        return Path(seed["db_path"]).expanduser().resolve()

    project = seed.get("project_name") or seed.get("repo_url") or "repo"
    cve = seed.get("cve_id") or "unknown_cve"
    return (paths.codeql_db_dir / f"{_slugify(str(project))}__{_slugify(str(cve))}").resolve()


def _resolve_baseline_query(paths: SeedPaths) -> Optional[Path]:
    """
    Baseline query/suite path resolution strategy (V1):
    - If env BASELINE_QUERY_PATH is set: use it (absolute or relative)
    - Else if queries/baseline.qls exists: use it
    - Else: return None (node will log SKIP baseline analyze)
    """
    env_path = os.getenv("BASELINE_QUERY_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()

    fallback_suite = paths.queries_dir / "baseline.qls"
    if fallback_suite.exists():
        return fallback_suite.resolve()

    return None


def normalize_seed_case(workspace_dir: str, seed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a normalized seed-case dict that downstream nodes can consume
    without guessing keys.
    """
    paths = default_seed_paths(workspace_dir)

    # minimal required keys
    required = ["project_name", "cve_id", "cwe_id", "repo_url", "vulnerable_file", "target_function"]
    missing = [k for k in required if not seed.get(k)]
    if missing:
        raise SeedMaterialError(f"Seed is missing required fields: {missing}. seed={seed}")

    repo_path = _resolve_repo_path(paths, seed)
    db_path = _resolve_db_path(paths, seed)

    baseline_query = _resolve_baseline_query(paths)

    normalized: Dict[str, Any] = {
        "project_name": seed["project_name"],
        "cve_id": seed["cve_id"],
        "cwe_id": seed["cwe_id"],
        "repo_url": seed["repo_url"],
        "vuln_commit_sha": seed.get("vuln_commit_sha", ""),
        "fix_commit_sha": seed.get("fix_commit_sha", ""),
        "patch_url": seed.get("patch_url", ""),
        "build_command": seed.get("build_command", ""),
        "description": seed.get("description", ""),

        # for source reading / anchoring
        "vulnerable_file": seed["vulnerable_file"],
        "target_function": seed["target_function"],
        # optional, allow later extension
        "vuln_line_hint": seed.get("vuln_line_hint"),

        # local material locations (V1 expects user prepared them)
        "repo_path": str(repo_path),
        "db_path": str(db_path),

        # analysis entry points
        "baseline_query_path": str(baseline_query) if baseline_query else None,
        # V1: regression defaults to same suite; later can be replaced with your templated suite
        "regression_query_path": str(baseline_query) if baseline_query else None,

        # optional CodeQL search paths (project-controlled)
        "codeql_search_paths": seed.get("codeql_search_paths", []),
    }
    return normalized


def load_and_normalize_seeds(workspace_dir: str) -> List[Dict[str, Any]]:
    paths = default_seed_paths(workspace_dir)
    raw = _load_json_array(paths.seeds_json)
    return [normalize_seed_case(workspace_dir, s) for s in raw]
