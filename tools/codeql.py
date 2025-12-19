"""
Lightweight CodeQL CLI wrapper.
Enhanced with robust DB creation, custom build commands, and finalization logic.
"""

import os
import shutil
import subprocess
import logging
from dataclasses import dataclass
from typing import List, Optional

from app.core.loader import get_str_env

logger = logging.getLogger(__name__)

def _resolve_codeql_path() -> str:
    """
    Resolve the path to the CodeQL executable from environment variables or PATH.
    """
    env_path = get_str_env("CODEQL_CLI_PATH", "")
    if env_path:
        # Allow pointing to either the binary or its parent directory
        if os.path.isdir(env_path):
            candidate = os.path.join(env_path, "codeql")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return env_path
    found = shutil.which("codeql")
    if found:
        return found
    return "codeql"


@dataclass
class CodeQLConfig:
    codeql_cli_path: str = _resolve_codeql_path()
    database_root: str = get_str_env("CODEQL_DATABASE_ROOT", ".codeql/db")
    search_path: Optional[str] = get_str_env("CODEQL_SEARCH_PATH", "data/ql_library")


class CodeQLRunner:
    def __init__(self, config: CodeQLConfig | None = None) -> None:
        self.config = config or CodeQLConfig()

    def _run(self, args: List[str]) -> subprocess.CompletedProcess:
        """
        Internal helper to execute subprocess commands.
        """
        cmd = [self.config.codeql_cli_path] + args
        # CodeQL writes logs to stderr and results to file (when --output is used).
        # We capture both for debugging purposes.
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def finalize_db(self, db_path: str) -> subprocess.CompletedProcess:
        """
        Attempts to finalize a database. 
        Useful if a previous creation was interrupted or to verify integrity.
        """
        return self._run(["database", "finalize", db_path])

    def create_db(
        self, 
        repo_path: str, 
        language: str = "cpp", 
        db_name: str | None = None, 
        command: str | None = None
    ) -> subprocess.CompletedProcess:
        """
        Creates a CodeQL database for the given repository.
        
        Args:
            repo_path: Path to the source code.
            language: Target language (e.g., 'cpp', 'java').
            db_name: Optional custom name for the DB directory.
            command: Explicit build command (e.g. "./configure && make"). 
                     Crucial for C/C++ projects where autobuild often fails.
        """
        db_path = os.path.join(self.config.database_root, db_name or f"{language}-db")
        
        # [Step 1] Robust Reuse Logic (Check - Verify - Rebuild)
        if os.path.exists(db_path):
            # Check for language-specific subdirectory (e.g., "db-cpp")
            # This confirms the extractor actually ran.
            expected_lang_dir = os.path.join(db_path, f"db-{language}")
            
            if os.path.exists(expected_lang_dir):
                logger.info(f"Database struct valid at {db_path}. Verifying integrity...")
                # Try to finalize it just in case it's incomplete
                finalize_result = self.finalize_db(db_path)
                
                if finalize_result.returncode == 0:
                    logger.info("Database is valid/finalized. Reusing.")
                    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            
            # If structure missing or finalize failed, it's corrupt. Nuke it.
            logger.warning(f"❌ Existing database at {db_path} is incomplete or corrupt. Recreating...")
            try:
                shutil.rmtree(db_path, ignore_errors=True)
            except Exception as e:
                return subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", 
                    stderr=f"Failed to remove corrupt database: {e}"
                )

        # Ensure parent directory exists
        os.makedirs(self.config.database_root, exist_ok=True)

        # [Step 2] Construct Creation Command
        cmd_args = [
            "database", "create", db_path, 
            f"--language={language}", 
            f"--source-root={repo_path}",
            "--overwrite" # Force overwrite if directory exists but is empty
        ]
        
        # [Fix] Inject build command if provided
        if command:
            cmd_args.append(f'--command={command}')
        
        logger.info(f"🚀 Creating DB with command: {' '.join(cmd_args)}")
        result = self._run(cmd_args)
        
        # [Step 3] Failure Cleanup
        # If returncode != 0 OR the language dir wasn't created, it failed.
        expected_lang_dir = os.path.join(db_path, f"db-{language}")
        if result.returncode != 0 or not os.path.exists(expected_lang_dir):
            logger.error(f"❌ Database creation failed. Cleaning up {db_path}")
            if result.stderr:
                logger.error(f"CodeQL Error: {result.stderr}")
            
            # Remove partial DB to prevent "needs to be finalized" errors next time
            shutil.rmtree(db_path, ignore_errors=True)
            
            # Ensure we return a failure code even if CodeQL exit code was 0 but dir is missing
            if result.returncode == 0:
                result.returncode = 1
                result.stderr = "Database created but language directory is missing (Build failed?)"
            
        return result

    def run_query(self, db_path: str, query_path: str, output_path: str) -> subprocess.CompletedProcess:
        """
        Runs a CodeQL query against a database.
        
        Args:
            db_path: Path to the CodeQL database.
            query_path: Path to the .ql query file.
            output_path: File path where the SARIF results should be saved.
        """
        # Paranoid check: if internal folder missing, try finalize before running
        if os.path.exists(db_path) and not os.path.exists(os.path.join(db_path, "db-cpp")): 
             self.finalize_db(db_path)

        args = [
            "database", 
            "analyze", 
            db_path, 
            query_path, 
            "--format=sarif-latest", 
            f"--output={output_path}", # [Fix] Mandatory option for SARIF output
            "--rerun" # [Fix] Force re-analysis even if cache exists
        ]
        
        if self.config.search_path:
            args.append(f"--search-path={self.config.search_path}")
            
        return self._run(args)