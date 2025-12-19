import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from app.core.loader import get_int_env, get_str_env
from app.rag.retriever import Chunk, Document, Resource, Retriever

logger = logging.getLogger(__name__)


class LocalChromaProvider(Retriever):
    """
    Local Chroma-based retriever.
    - Loads documents from data/knowledge_base (default) and builds a persistent Chroma index.
    - Uses SentenceTransformer embeddings for offline friendliness.
    """

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parent.parent.parent
        self.knowledge_dir = Path(
            get_str_env(
                "LOCAL_CHROMA_KB_DIR", project_root / "data" / "knowledge_base"
            )
        )
        self.persist_dir = Path(
            get_str_env("LOCAL_CHROMA_PERSIST_DIR", project_root / "data" / "chroma_db")
        )
        self.embedding_model_name = get_str_env(
            "LOCAL_CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.top_k = get_int_env("LOCAL_CHROMA_TOP_K", 8)
        self.chunk_size = get_int_env("LOCAL_CHROMA_CHUNK_SIZE", 800)

        self.embedding = SentenceTransformerEmbeddings(
            model_name=self.embedding_model_name
        )
        self.vector_store: Optional[Chroma] = None
        self.doc_index: Dict[str, Dict[str, str]] = {}

    def _load_files(self) -> None:
        if not self.knowledge_dir.exists():
            logger.warning("Knowledge base directory not found: %s", self.knowledge_dir)
            return
        texts: List[str] = []
        metadatas: List[Dict[str, str]] = []
        for file in self.knowledge_dir.glob("**/*"):
            if not file.is_file():
                continue
            if file.suffix.lower() not in {".md", ".txt", ".json", ".rst"}:
                continue
            try:
                content = file.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning("Failed to read %s: %s", file, e)
                continue
            if not content.strip():
                continue
            chunks = self._split_content(content)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{file.name}-{idx}"
                texts.append(chunk)
                metadatas.append(
                    {
                        "id": doc_id,
                        "source": str(file.relative_to(self.knowledge_dir)),
                        "title": file.name,
                    }
                )
                self.doc_index[doc_id] = metadatas[-1]

        if texts:
            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding,
                metadatas=metadatas,
                persist_directory=str(self.persist_dir),
            )
            self.vector_store.persist()
        else:
            logger.info("No documents loaded into LocalChromaProvider.")

    def _ensure_vector_store(self) -> None:
        if self.vector_store:
            return
        if self.persist_dir.exists() and any(self.persist_dir.iterdir()):
            self.vector_store = Chroma(
                embedding_function=self.embedding,
                persist_directory=str(self.persist_dir),
            )
            # Attempt to rebuild doc_index from stored metadatas
            try:
                for doc in self.vector_store.get(include=["metadatas"]).get(
                    "metadatas", []
                ):
                    if doc and isinstance(doc, dict) and "id" in doc:
                        self.doc_index[doc["id"]] = doc
            except Exception:
                pass
        else:
            self._load_files()

    def _split_content(self, content: str) -> List[str]:
        if len(content) <= self.chunk_size:
            return [content]
        chunks: List[str] = []
        paragraphs = content.split("\n\n")
        current = ""
        for para in paragraphs:
            if len(current) + len(para) <= self.chunk_size:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = para + "\n\n"
        if current:
            chunks.append(current.strip())
        return chunks

    def list_resources(self, query: str | None = None) -> List[Resource]:
        self._ensure_vector_store()
        resources: List[Resource] = []
        for doc_id, meta in self.doc_index.items():
            resources.append(
                Resource(
                    uri=f"chroma://{meta.get('source','')}",
                    title=meta.get("title", doc_id),
                    description=meta.get("source", ""),
                )
            )
        return resources

    def query_relevant_documents(
        self, query: str, resources: List[Resource] = []
    ) -> List[Document]:
        self._ensure_vector_store()
        if not self.vector_store:
            return []
        results = self.vector_store.similarity_search_with_score(
            query, k=self.top_k
        )
        docs: Dict[str, Document] = {}
        for doc, score in results:
            meta = doc.metadata or {}
            doc_id = meta.get("id") or meta.get("source") or doc.page_content[:20]
            title = meta.get("title")
            uri = meta.get("source")
            if resources:
                if not any(r.uri.endswith(uri or "") for r in resources):
                    continue
            if doc_id not in docs:
                docs[doc_id] = Document(id=doc_id, url=uri, title=title, chunks=[])
            docs[doc_id].chunks.append(
                Chunk(content=doc.page_content, similarity=score)
            )
        return list(docs.values())
