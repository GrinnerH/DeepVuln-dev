import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.core.tools import SELECTED_SEARCH_ENGINE, SearchEngine
from app.core.loader import load_yaml_config
from app.tools.decorators import create_logged_tool

logger = logging.getLogger(__name__)


def get_search_config():
    config = load_yaml_config("conf.yaml")
    return config.get("SEARCH_ENGINE", {})


class SimpleSearchInput(BaseModel):
    query: str = Field(description="search query to look up")


class SimpleSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Web search placeholder tool"
    args_schema: type[BaseModel] = SimpleSearchInput
    max_results: int = 5
    site: Optional[str] = None

    def _run(self, query: str) -> str:
        return f"[web_search stub] query={query} max_results={self.max_results} site={self.site or ''}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


LoggedSimpleSearchTool = create_logged_tool(SimpleSearchTool)


def get_web_search_tool(max_search_results: int):
    """
    DeerFlow-compatible factory. Real engines can be added later; currently returns
    a logged stub to keep dependencies light while preserving interface.
    """
    search_config = get_search_config()
    site = search_config.get("site")
    if SELECTED_SEARCH_ENGINE in {
        SearchEngine.TAVILY.value,
        SearchEngine.INFOQUEST.value,
        SearchEngine.DUCKDUCKGO.value,
        SearchEngine.BRAVE_SEARCH.value,
        SearchEngine.ARXIV.value,
        SearchEngine.SEARX.value,
        SearchEngine.WIKIPEDIA.value,
    }:
        return LoggedSimpleSearchTool(max_results=max_search_results, site=site)
    raise ValueError(f"Unsupported search engine: {SELECTED_SEARCH_ENGINE}")
