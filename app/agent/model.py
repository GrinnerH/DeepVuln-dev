"""
Model factory for DeepVuln 2.0 (LangGraph v1.0 compatible).

Responsibilities:
- Initialize chat models as LangChain Runnables
- Support optional tool binding (e.g. MCP / RAGFlow)
- Support optional structured output (Pydantic models)

This module MUST NOT:
- create agents
- control tool loops
- manage prompts
- mutate graph state
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Type

from langchain_core.language_models.chat import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_dev_utils.chat_models import (
    register_model_provider,
    load_chat_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_base_chat_model() -> BaseChatModel:
    """
    Load a chat model using env configuration.

    Required env:
      - OPENAI_API_KEY
    Optional env:
      - OPENAI_MODEL
      - OPENAI_BASE_URL / OPENAI_API_BASE / LITELLM_BASE_URL

    Uses FakeChatModel if explicitly requested via DEV_FAKE_LLM=1.
    """

    if os.getenv("DEV_FAKE_LLM") == "1":
        logger.warning("Using FakeChatModel (DEV_FAKE_LLM=1)")
        return FakeChatModel()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("LITELLM_BASE_URL")
    )

    # Register a provider alias so load_chat_model works uniformly
    register_model_provider(
        provider_name="DMXAPI",
        chat_model=FakeChatModel,
        base_url=base_url,
    )

    try:
        logger.info(
            "Loading chat model",
            extra={
                "model": model_name,
                "base_url": base_url,
            },
        )
        return load_chat_model(f"DMXAPI:{model_name}")
    except Exception as exc:
        logger.exception("Failed to load chat model", exc_info=exc)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model() -> BaseChatModel:
    """
    Build a bare chat model (Runnable).

    This is the lowest-level entry point.
    Nodes may further wrap it with tools: .bind_tools(...)
    or structured output: .with_structured_output(...)
    """
    return _load_base_chat_model()


def build_runnable(
    *,
    tools: Optional[Iterable[BaseTool]] = None,
    structured_schema: Optional[Type] = None,
) -> Runnable:
    """
    Build a LangGraph-compatible Runnable.

    Parameters
    ----------
    tools:
        Optional iterable of LangChain tools (e.g. MCP, RAGFlow).
        Tools are bound but NOT auto-invoked.
    structured_schema:
        Optional Pydantic model for structured output.
        Uses .with_structured_output(schema).

    Returns
    -------
    Runnable
        A runnable that supports invoke / ainvoke.
    """

    model = build_model()

    if tools:
        model = model.bind_tools(list(tools))
        logger.debug("Bound tools to model", extra={"tools": [t.name for t in tools]})

    if structured_schema:
        model = model.with_structured_output(structured_schema)
        logger.debug(
            "Enabled structured output",
            extra={"schema": structured_schema.__name__},
        )

    return model


def build_chat_model(
    model_name: Optional[str] = None,
    tools: Optional[Iterable[BaseTool]] = None,
    structured_schema: Optional[Type] = None,
) -> Runnable:
    """
    Backward-compatible alias for older node code.
    model_name is accepted for compatibility; actual selection is via env OPENAI_MODEL.
    """
    return build_runnable(tools=tools, structured_schema=structured_schema)