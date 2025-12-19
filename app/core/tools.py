# Tool-related enums and env selections aligned with DeerFlow.

import enum
import os

from dotenv import load_dotenv

load_dotenv()


class SearchEngine(enum.Enum):
    TAVILY = "tavily"
    INFOQUEST = "infoquest"
    DUCKDUCKGO = "duckduckgo"
    BRAVE_SEARCH = "brave_search"
    ARXIV = "arxiv"
    SEARX = "searx"
    WIKIPEDIA = "wikipedia"


class CrawlerEngine(enum.Enum):
    JINA = "jina"
    INFOQUEST = "infoquest"


SELECTED_SEARCH_ENGINE = os.getenv("SEARCH_API", SearchEngine.TAVILY.value)


class RAGProvider(enum.Enum):
    DIFY = "dify"
    RAGFLOW = "ragflow"
    VIKINGDB_KNOWLEDGE_BASE = "vikingdb_knowledge_base"
    MOI = "moi"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    LOCAL_CHROMA = "local_chroma"


SELECTED_RAG_PROVIDER = os.getenv("RAG_PROVIDER")
