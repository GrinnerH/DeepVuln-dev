from __future__ import annotations

"""Lightweight CodeQL "MCP" client.

This is intentionally *not* a full MCP/LangChain integration yet. It is a
minimal, controllable facade around the CodeQL CLI so the project can be made
end-to-end functional quickly.

Later you can replace this module with a real MCP transport (e.g., RAGFlow MCP
for docs + a CodeQL MCP server for CLI actions) without rewriting the LangGraph
nodes.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class CodeQLExecutionError(RuntimeError):
    """Raised when a CodeQL CLI command fails."""


class CodeQLMCPError(RuntimeError):
    pass


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

    # Fallback to PATH
    from shutil import which

    return which("codeql")


def _run(cmd: List[str], cwd: Optional[str] = None, timeout_s: int = 60 * 30) -> Dict[str, Any]:
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
        )
    except subprocess.TimeoutExpired as e:
        raise CodeQLMCPError(f"CodeQL command timed out: {cmd}") from e

    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


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

    # --- Public API ---

    def analyze_database(
        self,
        *,
        db_path: str,
        query_or_suite_path: str,
        output_sarif_path: str,
        additional_search_paths: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run `codeql database analyze`.

        If CodeQL is unavailable, returns a stub response with returncode=127.
        """
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
            self.codeql_path,  # type: ignore[arg-type]
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

        result = _run(cmd, cwd=cwd)
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
        """Run `codeql query compile` for a single .ql file."""
        if not self.available:
            return {
                "ok": False,
                "returncode": 127,
                "stderr": "codeql executable not found (set CODEQL_CLI_PATH)",
                "stdout": "",
                "cmd": [],
            }

        cmd: List[str] = [self.codeql_path, "query", "compile", str(query_path)]  # type: ignore[list-item]
        if additional_search_paths:
            for p in additional_search_paths:
                cmd.append(f"--search-path={p}")
        if extra_args:
            cmd.extend(extra_args)

        result = _run(cmd, cwd=cwd)
        result.update({"ok": result["returncode"] == 0})
        return result
