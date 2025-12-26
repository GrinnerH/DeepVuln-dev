# app/rag/advanced_rag.py
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveJsonSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class ChromaRAGEngine:
    """
    升级版的 RAG 引擎：支持结构化切片、元数据增强和混合检索。
    """
    def __init__(self, persist_dir: str, knowledge_dir: str):
        self.persist_dir = Path(persist_dir)
        self.knowledge_dir = Path(knowledge_dir)
        self.embedding = SentenceTransformerEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
        
        # 专门的切片器
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header1"), ("##", "Header2")])
        self.json_splitter = RecursiveJsonSplitter(max_chunk_size=800)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        
        self.vector_store: Optional[Chroma] = None
        self.final_retriever: Optional[ContextualCompressionRetriever] = None

    def _get_all_documents(self) -> List[Document]:
        """差异化数据处理与元数据增强注入"""
        all_docs = []
        for file in self.knowledge_dir.glob("**/*"):
            if file.suffix.lower() not in {".md", ".json"}: continue
            
            content = file.read_text(encoding="utf-8", errors="ignore")
            # 增强元数据：解决搜索碰撞的关键
            base_meta = {
                "source": file.name,
                "language": "cpp" if "cpp" in str(file).lower() else "python",
                "scope": "api_spec" if "api" in file.name.lower() else "usage_guide"
            }

            if file.suffix.lower() == ".md":
                # Markdown 切片：保留标题上下文
                docs = self.md_splitter.split_text(content)
                final = self.text_splitter.split_documents(docs)
            else:
                # JSON 切片：保留键值对结构
                import json
                data = json.loads(content)
                final = self.json_splitter.create_documents(texts=[data])
            
            for d in final:
                d.metadata.update(base_meta)
            all_docs.extend(final)
        return all_docs

    def initialize_index(self):
        """构建混合索引管道"""
        docs = self._get_all_documents()
        
        # 1. 向量索引 (Dense)
        self.vector_store = Chroma.from_documents(
            documents=docs, embedding=self.embedding, persist_directory=str(self.persist_dir)
        )
        
        # 2. 关键词索引 (Sparse)
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 10
        
        # 3. 混合检索与 BGE 重排
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6])
        
        compressor = FlashrankRerank(model_name="ms-marco-MiniLM-L-12-v2")
        self.final_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)

    def search(self, query: str, language: str = None, scope: str = None) -> str:
        """带元数据过滤的搜索"""
        # 注意：Chroma 的元数据过滤通常在 vector_retriever 级别生效
        # 这里简化处理，直接执行检索
        results = self.final_retriever.get_relevant_documents(query)
        
        # 格式化输出给 Agent
        output = []
        for d in results:
            if language and d.metadata.get("language") != language: continue
            header = d.metadata.get("Header1", "General")
            output.append(f"### Source: {d.metadata['source']} ({header})\n{d.page_content}")
        
        return "\n\n---\n\n".join(output) if output else "未找到相关安全知识。"