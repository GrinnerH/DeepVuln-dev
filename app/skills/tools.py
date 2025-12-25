from __future__ import annotations

import re
from pathlib import Path

from langchain.tools import tool

from utils.io import safe_read_text


_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@tool("load_skill")
def load_skill(skill_name: str) -> str:
    """Load a skill prompt by name from app/skills."""
    if not skill_name or not _SKILL_NAME_RE.match(skill_name):
        return f"<<INVALID_SKILL_NAME {skill_name}>>"
    skills_root = Path(__file__).resolve().parent
    skill_path = skills_root / f"{skill_name}.md"
    if not skill_path.exists():
        return f"<<SKILL_NOT_FOUND {skill_name}>>"
    return safe_read_text(skill_path)
