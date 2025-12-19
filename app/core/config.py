import logging
import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from app.core.loader import get_int_env, get_str_env
from app.core.tools import SELECTED_RAG_PROVIDER, RAGProvider
from app.rag.retriever import Resource

load_dotenv()

logger = logging.getLogger(__name__)


def get_recursion_limit(default: int = 25) -> int:
    """Align with DeerFlow: read AGENT_RECURSION_LIMIT and validate."""
    env_value_str = get_str_env("AGENT_RECURSION_LIMIT", str(default))
    parsed_limit = get_int_env("AGENT_RECURSION_LIMIT", default)
    if parsed_limit > 0:
        logger.info("Recursion limit set to: %s", parsed_limit)
        return parsed_limit
    logger.warning(
        "AGENT_RECURSION_LIMIT value '%s' (parsed as %s) is not positive. Using default %s.",
        env_value_str,
        parsed_limit,
        default,
    )
    return default


@dataclass(kw_only=True)
class Configuration:
    """
    Mirrors DeerFlow's Configuration for compatibility with env/configurable inputs.
    """

    resources: list[Resource] = field(default_factory=list)
    max_plan_iterations: int = 1
    max_step_num: int = 3
    max_search_results: int = 3
    mcp_settings: dict | None = None
    report_style: str = "academic"
    enable_deep_thinking: bool = False
    enforce_web_search: bool = False
    enforce_researcher_search: bool = True
    interrupt_before_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})


# Alias used by agents/graph for clarity
GraphConfig = Configuration
