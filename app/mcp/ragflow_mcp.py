# app/mcp/ragflow_mcp.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict

from langchain.tools import tool
"""
Placeholder facade for RAGFlow MCP usage.

You indicated you'll reuse DeerFlow's RAGFlow MCP integration.
Keep this wrapper so nodes can call `search()` without coupling to the underlying transport.
"""


@dataclass(frozen=True)
class RAGFlowMCPClient:
    endpoint: str

    @staticmethod
    def from_env() -> "RAGFlowMCPClient":
        # Customize env var names as your MCP layer defines.
        endpoint = os.getenv("RAGFLOW_MCP_ENDPOINT", "").strip()
        return RAGFlowMCPClient(endpoint=endpoint)

    def search(self, query: str, top_k: int = 5, **kwargs: Any) -> Dict[str, Any]:
        # Replace with real MCP call
        return {
            "status": "stub",
            "query": query,
            "top_k": top_k,
            "citation_refs": [],
            "results": [],
        }


@tool
def ragflow_search_tool(query: str, top_k: int = 5) -> str:
    """Search RAGFlow MCP for relevant documents."""
    client = RAGFlowMCPClient.from_env()
    res = client.search(query=query, top_k=top_k)
    return json.dumps(res, ensure_ascii=False)
