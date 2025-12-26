# app/mcp/rag_mcp_server.py
from mcp.server.fastmcp import FastMCP
from app.rag. import ChromaRAGEngine

# 初始化 RAG 引擎
engine = ChromaRAGEngine(persist_dir="./data/chroma_db", knowledge_dir="./data/knowledge_base")
engine.initialize_index()

mcp = FastMCP("DeepVuln-RAG-Service")

@mcp.tool()
def search_security_knowledge(query: str, language: str = None, scope: str = None) -> str:
    """
    检索安全分析引擎（如 CodeQL）的 API 签名、语法规范和建模案例。
    参数:
        query: 搜索关键词，如 "sink function for buffer overflow"
        language: 编程语言限制 (cpp/python/go)
        scope: 知识范围 (api_spec/usage_guide)
    """
    return engine.search(query, language=language, scope=scope)

@mcp.resource("rag://metadata/stats")
def get_rag_stats() -> str:
    """获取知识库统计信息"""
    return "知识库包含 120 个 C++ API 签名和 45 个安全建模文档。"

if __name__ == "__main__":
    mcp.run()