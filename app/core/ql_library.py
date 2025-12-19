import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional


class QLLibrary:
    """
    Manage persisted QL scripts.
    
    Structure:
    data/ql_library/{language}/{cwe}/
        manifest.json               # Global tracking for best merged queries
        v1_*.ql, verified_best.ql   # Global/Merged versions
        
        case_{sanitized_id}/        # Per-case workspace
            source.ql
            sink.ql
            config.ql
    """

    def __init__(self, language: str, cwe_id: str, root: str = "data/ql_library") -> None:
        self.language = language or "cpp"
        self.cwe_id = cwe_id
        self.root = Path(root)
        self.base_dir = self.root / self.language / self.cwe_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "manifest.json"
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        if not self.manifest_path.exists():
            data = {"best_query": "", "history": []}
            self.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"best_query": "", "history": []}

    def _save_manifest(self, data: dict) -> None:
        self.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _next_version_name(self) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        manifest = self._load_manifest()
        version = len(manifest.get("history", [])) + 1
        return f"v{version}_{ts}.ql"

    # --- Global / Merged Query Management ---

    def write_candidate(self, ql_code: str) -> Tuple[str, Path]:
        """Write a candidate QL file into the root of the library (for merged/global scripts)."""
        name = self._next_version_name()
        path = self.base_dir / name
        path.write_text(ql_code, encoding="utf-8")
        return name, path

    def update_manifest(
        self,
        filename: str,
        compile_success: bool,
        verified_cases: List[str],
        failed_cases: List[str],
    ) -> None:
        manifest = self._load_manifest()
        history = manifest.get("history", [])
        entry = {
            "filename": filename,
            "timestamp": datetime.utcnow().isoformat(),
            "compile_success": compile_success,
            "verified_cases": verified_cases,
            "failed_cases": failed_cases,
        }
        history.append(entry)
        manifest["history"] = history

        # Choose best: prefer more verified, fewer failed, compile_success
        best = manifest.get("best_query", "")
        best_score = (-1, 9999)
        if best:
            for h in history:
                if h["filename"] == best:
                    best_score = (len(h.get("verified_cases", [])), len(h.get("failed_cases", [])))
                    break
        current_score = (len(verified_cases), len(failed_cases))
        if compile_success and current_score > best_score:
            manifest["best_query"] = filename

        self._save_manifest(manifest)

    def get_best_query(self, reference_cases: List[dict]) -> Tuple[str | None, List[str]]:
        """
        Return best global query content and missing reference ids.
        """
        manifest = self._load_manifest()
        best = manifest.get("best_query")
        if not best:
            return None, []
        best_path = self.base_dir / best
        if not best_path.exists():
            return None, []
        ref_ids = [c.get("vuln_commit_sha") or c.get("repo_url") for c in reference_cases or []]
        ref_ids = [r for r in ref_ids if r]

        # Find history entry for best
        history = manifest.get("history", [])
        best_entry = next((h for h in history if h.get("filename") == best), None)
        verified = set(best_entry.get("verified_cases", [])) if best_entry else set()
        missing = [r for r in ref_ids if r not in verified]

        return best_path.read_text(encoding="utf-8"), missing

    # --- Per-Case Fragment Management (New Feature) ---

    def _get_case_dir(self, case_ref: dict) -> Path:
        """
        Generate a unique, safe directory path for a specific reference case.
        """
        # Prioritize commit SHA, fallback to repo URL, then generic fallback
        ref_id = case_ref.get("vuln_commit_sha") or case_ref.get("repo_url") or "unknown_case"
        
        # Sanitize ID to be filesystem safe (alphanumeric + underscore only)
        # Taking last 16 chars usually enough to distinguish commits/hashes
        safe_suffix = "".join(c if c.isalnum() else "_" for c in str(ref_id))[-24:]
        
        case_dir_name = f"case_{safe_suffix}"
        case_path = self.base_dir / case_dir_name
        case_path.mkdir(parents=True, exist_ok=True)
        return case_path

    def write_case_fragment(self, case_ref: dict, phase: str, ql_code: str) -> Path:
        """
        Save a QL fragment for a specific case and phase.
        phase: 'source', 'sink', 'config' (or 'final')
        """
        case_dir = self._get_case_dir(case_ref)
        filename = f"{phase}.ql"
        path = case_dir / filename
        path.write_text(ql_code, encoding="utf-8")
        return path

    def get_case_fragment(self, case_ref: dict, phase: str) -> str:
        """
        Retrieve a QL fragment for a specific case.
        Returns empty string if not found.
        """
        case_dir = self._get_case_dir(case_ref)
        path = case_dir / f"{phase}.ql"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""