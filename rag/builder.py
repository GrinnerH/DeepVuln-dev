from app.core.tools import RAGProvider, SELECTED_RAG_PROVIDER
from app.rag.local_chroma import LocalChromaProvider
from app.rag.milvus import MilvusProvider
from app.rag.qdrant import QdrantProvider
from app.rag.ragflow import RAGFlowProvider
from app.rag.retriever import Retriever


def build_retriever() -> Retriever | None:
    """
    Factory aligned with DeerFlow, with local_chroma as default.
    """
    if SELECTED_RAG_PROVIDER == RAGProvider.RAGFLOW.value:
        return RAGFlowProvider()
    if SELECTED_RAG_PROVIDER == RAGProvider.MILVUS.value:
        return MilvusProvider()
    if SELECTED_RAG_PROVIDER == RAGProvider.QDRANT.value:
        return QdrantProvider()
    if SELECTED_RAG_PROVIDER and SELECTED_RAG_PROVIDER not in {
        RAGProvider.LOCAL_CHROMA.value,
        "",
        None,
    }:
        raise ValueError(f"Unsupported RAG provider: {SELECTED_RAG_PROVIDER}")
    return LocalChromaProvider()
