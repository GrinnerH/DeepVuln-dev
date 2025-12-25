from __future__ import annotations

"""Lightweight CodeQL "MCP" client.

This module is intentionally *not* a full MCP/LangChain integration yet.
It provides a minimal, controllable facade around the CodeQL CLI.

Design goals:
  - Single entry-point for all CodeQL CLI interactions.
  - Structured return objects for auditability (stdout/stderr/cmd).
  - "Good enough" robustness for real-world C/C++ builds (Step A automation).

Later you can replace this module with a real MCP transport (e.g., a dedicated
CodeQL MCP server) without rewriting the LangGraph nodes.
"""

import os
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.tools import tool


class CodeQLExecutionError(RuntimeError):
    """Raised when a CodeQL CLI command fails."""


class CodeQLMCPError(RuntimeError):
    """Raised when a CodeQL CLI interaction fails for non-CLI reasons."""


def _resolve_codeql_executable() -> Optional[str]:
    """Resolve the `codeql` executable path.

    Priority:
      1) CODEQL_CLI_PATH env var (directory containing codeql)
      2) PATH lookup
    """

    cli_dir = os.getenv("CODEQL_CLI_PATH", "").strip()
    if cli_dir:
        candidate = Path(cli_dir) / "codeql"
        if os.name == "nt":
            candidate = candidate.with_suffix(".exe")
        if candidate.exists():
            return str(candidate)

    from shutil import which

    return which("codeql")


def _default_env() -> Dict[str, str]:
    """Return an environment suited for CodeQL executions.

    Notes:
      - CodeQL uses CODEQL_JAVA_HOME if set.
      - Some large DB builds need higher memory; leave to user's environment.
    """

    env = dict(os.environ)
    # Ensure deterministic encoding handling across platforms.
    env.setdefault("LC_ALL", "C")
    env.setdefault("LANG", "C")
    return env


def _run(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    timeout_s: int = 60 * 60,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run a subprocess, capturing stdout/stderr for auditing."""

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            env=env or _default_env(),
        )
    except subprocess.TimeoutExpired as e:
        raise CodeQLMCPError(f"CodeQL command timed out: {cmd}") from e

    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _ensure_dir(path: str) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


@dataclass
class CodeQLMCPClient:
    """A small facade around the CodeQL CLI.

    Methods return structured dictionaries so LangGraph nodes can log and store
    evidence references.
    """

    codeql_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.codeql_path is None:
            self.codeql_path = _resolve_codeql_executable()

    @property
    def available(self) -> bool:
        return bool(self.codeql_path)

    # ---------------------------------------------------------------------
    # Database lifecycle (migrated from legacy tools/codeql.py for robustness)
    # ---------------------------------------------------------------------

    def create_database(
        self,
        *,
        source_root: str,
        db_path: str,
        language: str = "cpp",
        command: Optional[str] = None,
        overwrite: bool = False,
        threads: Optional[int] = None,
        ram_mb: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run `codeql database create`.

        This absorbs the "good enough" build-handling logic from the legacy
        implementation (overwrite, threads, env, and optional build command).
        """

        db_dir = Path(db_path)
        if db_dir.exists() and overwrite:
            # Danger: remove existing db dir.
            import shutil

            shutil.rmtree(db_dir)

        _ensure_dir(str(db_dir.parent))

        if not self.available:
            return {
                "ok": False,
                "returncode": 127,
                "stderr": "codeql executable not found (set CODEQL_CLI_PATH)",
                "stdout": "",
                "cmd": [],
                "db_path": str(db_dir),
            }

        cmd_list: List[str] = [
            self.codeql_path,  # type: ignore[list-item]
            "database",
            "create",
            str(db_dir),
            f"--language={language}",
            f"--source-root={str(Path(source_root))}",
        ]

        if overwrite:
            cmd_list.append("--overwrite")
        if threads:
            cmd_list.append(f"--threads={threads}")
        if ram_mb:
            cmd_list.append(f"--ram={ram_mb}")
        if command:
            # Use a shell command string; CodeQL accepts it directly.
            cmd_list.append(f"--command={command}")
        if extra_args:
            cmd_list.extend(extra_args)

        result = _run(cmd_list, cwd=cwd)
        result.update({"ok": result["returncode"] == 0, "db_path": str(db_dir)})
        return result

    def finalize_database(
        self,
        *,
        db_path: str,
        finalize_dataset: bool = True,
        extra_args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run `codeql database finalize`.

        Useful after DB creation in some environments to ensure dataset
        completeness.
        """

        if not self.available:
            return {
                "ok": False,
                "returncode": 127,
                "stderr": "codeql executable not found (set CODEQL_CLI_PATH)",
                "stdout": "",
                "cmd": [],
            }

        cmd_list: List[str] = [
            self.codeql_path,  # type: ignore[list-item]
            "database",
            "finalize",
            str(db_path),
        ]
        if finalize_dataset:
            cmd_list.append("--finalize-dataset")
        if extra_args:
            cmd_list.extend(extra_args)

        result = _run(cmd_list, cwd=cwd)
        result.update({"ok": result["returncode"] == 0})
        return result


    # -------------------------
    # Query/analyze operations
    # -------------------------

    def analyze_database(
        self,
        *,
        db_path: str,
        query_or_suite_path: str,
        output_sarif_path: str,
        additional_search_paths: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        timeout_s: int = 60 * 60,
    ) -> Dict[str, Any]:
        """Run `codeql database analyze`."""

        output = Path(output_sarif_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if not self.available:
            return {
                "ok": False,
                "returncode": 127,
                "stderr": "codeql executable not found (set CODEQL_CLI_PATH)",
                "stdout": "",
                "cmd": [],
                "output_sarif_path": str(output),
            }

        cmd: List[str] = [
            self.codeql_path,  # type: ignore[list-item]
            "database",
            "analyze",
            str(db_path),
            str(query_or_suite_path),
            "--format=sarif-latest",
            f"--output={str(output)}",
        ]
        if additional_search_paths:
            for p in additional_search_paths:
                cmd.append(f"--search-path={p}")
        if extra_args:
            cmd.extend(extra_args)

        result = _run(cmd, cwd=cwd, timeout_s=timeout_s)
        result.update({"ok": result["returncode"] == 0, "output_sarif_path": str(output)})
        return result

    def query_compile(
        self,
        *,
        query_path: str,
        additional_search_paths: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run `codeql query compile` for a single .ql/.qll file."""

        if not self.available:
            return {
                "ok": False,
                "returncode": 127,
                "stderr": "codeql executable not found (set CODEQL_CLI_PATH)",
                "stdout": "",
                "cmd": [],
            }

        cmd: List[str] = [
            self.codeql_path,  # type: ignore[list-item]
            "query",
            "compile",
            str(query_path),
        ]
        if additional_search_paths:
            for p in additional_search_paths:
                cmd.append(f"--search-path={p}")
        if extra_args:
            cmd.extend(extra_args)

        result = _run(cmd, cwd=cwd)
        result.update({"ok": result["returncode"] == 0})
        return result


@tool
def codeql_analyze_tool(
    db_path: str,
    query_path: str,
    output_sarif_path: str,
    search_paths: Optional[List[str]] = None,
) -> str:
    """Run CodeQL database analyze for diagnostic queries."""
    codeql = CodeQLMCPClient()
    res = codeql.analyze_database(
        db_path=db_path,
        query_or_suite_path=query_path,
        output_sarif_path=output_sarif_path,
        additional_search_paths=search_paths or None,
        extra_args=["--rerun"],
        cwd=None,
    )
    return json.dumps(res, ensure_ascii=False)
