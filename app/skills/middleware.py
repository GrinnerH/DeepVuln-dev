from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import logging

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from utils.io import safe_read_text
from app.skills.tools import load_skill


def _extract_description(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.lstrip("#").strip()
    return "No description available."


class SkillMiddleware(AgentMiddleware):
    """Inject available skill descriptions into the system prompt."""

    tools = [load_skill]

    def __init__(self) -> None:
        skills_root = Path(__file__).resolve().parent
        skills: List[Dict[str, str]] = []
        for path in sorted(skills_root.glob("*.md")):
            content = safe_read_text(path)
            skills.append(
                {
                    "name": path.stem,
                    "description": _extract_description(content),
                }
            )
        skills_list = [f"- **{s['name']}**: {s['description']}" for s in skills]
        self.skills_prompt = "\n".join(skills_list)
        logging.getLogger("deepvuln.v2").info(
            "[skills] middleware initialized skills=%s",
            [s["name"] for s in skills],
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        skills_addendum = (
            "\n\n## Available Skills\n\n"
            f"{self.skills_prompt}\n\n"
            "Use the load_skill tool when you need detailed information "
            "about handling a specific type of request."
        )
        logging.getLogger("deepvuln.v2").info(
            "[skills] injecting skills into system prompt chars=%d",
            len(skills_addendum),
        )
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)
