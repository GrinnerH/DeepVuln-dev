import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_git(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run git command with output capture; log on failure."""
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(
            "git command failed",
            extra={"cmd": cmd, "stdout": result.stdout, "stderr": result.stderr},
        )
    return result


def clone_or_update_repo(repo_url: str, commit_sha: str, dest_root: str = "data/reference_repos") -> str:
    """
    Clone the repo if missing; fetch and checkout the given commit.
    Returns local repo path.
    """
    dest_root_path = Path(dest_root)
    dest_root_path.mkdir(parents=True, exist_ok=True)
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    dest_path = dest_root_path / f"{repo_name}_{commit_sha[:8]}"

    # 如果目录已存在，视为已有缓存，直接复用并尝试切换到目标 commit（避免重复下载）
    if dest_path.exists():
        if commit_sha:
            checkout_result = _run_git(["git", "-C", str(dest_path), "checkout", commit_sha])
            if checkout_result.returncode != 0:
                logger.warning(
                    "Checkout failed on existing repo; keeping current revision.",
                    extra={"repo": str(dest_path), "commit": commit_sha, "stderr": checkout_result.stderr},
                )
        return str(dest_path.resolve())

    # Clone if missing
    clone_result = _run_git(["git", "clone", repo_url, str(dest_path)])
    if clone_result.returncode != 0:
        if dest_path.exists():
            logger.warning(
                "Clone reported failure but path exists; reusing existing path.",
                extra={"repo": repo_url, "path": str(dest_path), "stderr": clone_result.stderr},
            )
        else:
            raise RuntimeError(f"Failed to clone {repo_url}: {clone_result.stderr or clone_result.stdout}")

    # Checkout requested commit if provided
    if commit_sha:
        checkout_result = _run_git(["git", "-C", str(dest_path), "checkout", commit_sha])
        if checkout_result.returncode != 0:
            if dest_path.exists():
                logger.warning(
                    "Checkout failed; keeping current revision.",
                    extra={"repo": str(dest_path), "commit": commit_sha, "stderr": checkout_result.stderr},
                )
            else:
                raise RuntimeError(f"Failed to checkout {commit_sha} in {dest_path}: {checkout_result.stderr or checkout_result.stdout}")

    if not dest_path.exists():
        raise FileNotFoundError(f"Repository path not found after clone/update: {dest_path}")

    return str(dest_path.resolve())
