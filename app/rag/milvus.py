import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from langchain_milvus.vectorstores import Milvus as LangchainMilvus
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from app.core.loader import get_bool_env, get_int_env, get_str_env
from app.rag.retriever import Chunk, Document, Resource, Retriever

logger = logging.getLogger(__name__)


class DashscopeEmbeddings:
    """OpenAI-compatible embeddings wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        self._client: OpenAI = OpenAI(
            api_key=kwargs.get("api_key", ""), base_url=kwargs.get("base_url", "")
        )
        self._model: str = kwargs.get("model", "")
        self._encoding_format: str = kwargs.get("encoding_format", "float")

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Internal helper performing the embedding API call."""
        clean_texts = [t if isinstance(t, str) else str(t) for t in texts]
        if not clean_texts:
            return []
        resp = self._client.embeddings.create(
            model=self._model,
            input=clean_texts,
            encoding_format=self._encoding_format,
        )
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        """Return embedding for a given text."""
        embeddings = self._embed([text])
        return embeddings[0] if embeddings else []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for multiple documents (LangChain interface)."""
        return self._embed(texts)


class MilvusRetriever(Retriever):
    """Retriever implementation backed by a Milvus vector store."""

    def __init__(self) -> None:
        self.uri: str = get_str_env("MILVUS_URI", "http://localhost:19530")
        self.user: str = get_str_env("MILVUS_USER")
        self.password: str = get_str_env("MILVUS_PASSWORD")
        self.collection_name: str = get_str_env("MILVUS_COLLECTION", "documents")

        top_k_raw = get_str_env("MILVUS_TOP_K", "10")
        self.top_k: int = int(top_k_raw) if top_k_raw.isdigit() else 10

        self.vector_field: str = get_str_env("MILVUS_VECTOR_FIELD", "embedding")
        self.id_field: str = get_str_env("MILVUS_ID_FIELD", "id")
        self.content_field: str = get_str_env("MILVUS_CONTENT_FIELD", "content")
        self.title_field: str = get_str_env("MILVUS_TITLE_FIELD", "title")
        self.url_field: str = get_str_env("MILVUS_URL_FIELD", "url")
        self.metadata_field: str = get_str_env("MILVUS_METADATA_FIELD", "metadata")

        self.embedding_model = get_str_env("MILVUS_EMBEDDING_MODEL")
        self.embedding_api_key = get_str_env("MILVUS_EMBEDDING_API_KEY")
        self.embedding_base_url = get_str_env("MILVUS_EMBEDDING_BASE_URL")
        self.embedding_dim: int = self._get_embedding_dimension(self.embedding_model)
        self.embedding_provider = get_str_env("MILVUS_EMBEDDING_PROVIDER", "openai")

        self.auto_load_examples: bool = get_bool_env("MILVUS_AUTO_LOAD_EXAMPLES", True)
        self.examples_dir: str = get_str_env("MILVUS_EXAMPLES_DIR", "examples")
        self.chunk_size: int = get_int_env("MILVUS_CHUNK_SIZE", 4000)

        self._init_embedding_model()

        self.client: Any = None

    def _init_embedding_model(self) -> None:
        kwargs = {
            "api_key": self.embedding_api_key,
            "model": self.embedding_model,
            "base_url": self.embedding_base_url,
            "encoding_format": "float",
            "dimensions": self.embedding_dim,
        }
        if self.embedding_provider.lower() == "openai":
            self.embedding_model = OpenAIEmbeddings(**kwargs)
        elif self.embedding_provider.lower() == "dashscope":
            self.embedding_model = DashscopeEmbeddings(**kwargs)
        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.embedding_provider}. "
                "Supported providers: openai, dashscope"
            )

    def _get_embedding_dimension(self, model_name: str) -> int:
        embedding_dims = {
            "text-embedding-ada-002": 1536,
            "text-embedding-v4": 2048,
        }
        explicit_dim = get_int_env("MILVUS_EMBEDDING_DIM", 0)
        if explicit_dim > 0:
            return explicit_dim
        return embedding_dims.get(model_name, 1536)

    def _create_collection_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(
                name=self.id_field,
                dtype=DataType.VARCHAR,
                max_length=512,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name=self.vector_field,
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
            ),
            FieldSchema(
                name=self.content_field, dtype=DataType.VARCHAR, max_length=65535
            ),
            FieldSchema(name=self.title_field, dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name=self.url_field, dtype=DataType.VARCHAR, max_length=1024),
        ]
        return CollectionSchema(
            fields=fields,
            description=f"Collection for DeerFlow RAG documents: {self.collection_name}",
            enable_dynamic_field=True,
        )

    def _ensure_collection_exists(self) -> None:
        if self._is_milvus_lite():
            try:
                collections = self.client.list_collections()
                if self.collection_name not in collections:
                    schema = self._create_collection_schema()
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        schema=schema,
                        index_params={
                            "field_name": self.vector_field,
                            "index_type": "IVF_FLAT",
                            "metric_type": "IP",
                            "params": {"nlist": 1024},
                        },
                    )
                    logger.info("Created Milvus collection: %s", self.collection_name)
            except Exception as e:
                logger.warning("Could not ensure collection exists: %s", e)
        else:
            logger.warning("Could not ensure collection exists: %s", self.collection_name)

    def _load_example_files(self) -> None:
        try:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            examples_path = project_root / self.examples_dir

            if not examples_path.exists():
                logger.info("Examples directory not found: %s", examples_path)
                return

            logger.info("Loading example files from: %s", examples_path)
            md_files = list(examples_path.glob("*.md"))
            if not md_files:
                logger.info("No markdown files found in examples directory")
                return
            existing_docs = self._get_existing_document_ids()
            loaded_count = 0
            for md_file in md_files:
                doc_id = self._generate_doc_id(md_file)
                if doc_id in existing_docs:
                    continue
                try:
                    content = md_file.read_text(encoding="utf-8")
                    title = self._extract_title_from_markdown(content, md_file.name)
                    chunks = self._split_content(content)
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{doc_id}_chunk_{i}" if len(chunks) > 1 else doc_id
                        self._insert_document_chunk(
                            doc_id=chunk_id,
                            content=chunk,
                            title=title,
                            url=f"milvus://{self.collection_name}/{md_file.name}",
                            metadata={"source": "examples", "file": md_file.name},
                        )
                    loaded_count += 1
                    logger.debug("Loaded example markdown: %s", md_file.name)
                except Exception as e:
                    logger.warning("Error loading %s: %s", md_file.name, e)
            logger.info("Successfully loaded %d example files into Milvus", loaded_count)
        except Exception as e:
            logger.error("Error loading example files: %s", e)

    def _generate_doc_id(self, file_path: Path) -> str:
        file_stat = file_path.stat()
        content_hash = hashlib.md5(
            f"{file_path.name}_{file_stat.st_size}_{file_stat.st_mtime}".encode()
        ).hexdigest()[:8]
        return f"example_{file_path.stem}_{content_hash}"

    def _extract_title_from_markdown(self, content: str, filename: str) -> str:
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return filename.replace(".md", "").replace("_", " ").title()

    def _split_content(self, content: str) -> List[str]:
        if len(content) <= self.chunk_size:
            return [content]
        chunks = []
        paragraphs = content.split("\n\n")
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _get_existing_document_ids(self) -> Set[str]:
        try:
            if self._is_milvus_lite():
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter="",
                    output_fields=[self.id_field],
                    limit=10000,
                )
                return {
                    result.get(self.id_field, "")
                    for result in results
                    if result.get(self.id_field)
                }
            return set()
        except Exception:
            return set()

    def _insert_document_chunk(
        self, doc_id: str, content: str, title: str, url: str, metadata: Dict[str, Any]
    ) -> None:
        embedding = self._get_embedding(content)
        if self._is_milvus_lite():
            data = [
                {
                    self.id_field: doc_id,
                    self.vector_field: embedding,
                    self.content_field: content,
                    self.title_field: title,
                    self.url_field: url,
                    **metadata,
                }
            ]
            self.client.insert(collection_name=self.collection_name, data=data)
        else:
            self.client.add_texts(
                texts=[content],
                metadatas=[
                    {
                        self.id_field: doc_id,
                        self.title_field: title,
                        self.url_field: url,
                        **metadata,
                    }
                ],
            )

    def _connect(self) -> None:
        try:
            if self._is_milvus_lite():
                self.client = MilvusClient(self.uri)
                self._ensure_collection_exists()
            else:
                connection_args = {"uri": self.uri}
                if self.user:
                    connection_args["user"] = self.user
                if self.password:
                    connection_args["password"] = self.password
                self.client = LangchainMilvus(
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name,
                    connection_args=connection_args,
                    drop_old=False,
                )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")

    def _is_milvus_lite(self) -> bool:
        return self.uri.endswith(".db") or (
            not self.uri.startswith(("http://", "https://")) and "://" not in self.uri
        )

    def _get_embedding(self, text: str) -> List[float]:
        try:
            if not isinstance(text, str):
                raise ValueError(f"Text must be a string, got {type(text)}")
            if not text.strip():
                raise ValueError("Text cannot be empty or only whitespace")
            embeddings = self.embedding_model.embed_query(text=text.strip())
            if not isinstance(embeddings, list) or not embeddings:
                raise ValueError(f"Invalid embedding format: {type(embeddings)}")
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def list_resources(self, query: Optional[str] = None) -> List[Resource]:
        resources: List[Resource] = []
        if not self.client:
            try:
                self._connect()
            except Exception:
                return self._list_local_markdown_resources()
        try:
            if self._is_milvus_lite():
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter="source == 'examples'",
                    output_fields=[self.id_field, self.title_field, self.url_field],
                    limit=100,
                )
                for r in results:
                    resources.append(
                        Resource(
                            uri=r.get(self.url_field, "")
                            or f"milvus://{r.get(self.id_field, '')}",
                            title=r.get(self.title_field, "")
                            or r.get(self.id_field, "Unnamed"),
                            description="Stored Milvus document",
                        )
                    )
            else:
                docs: Iterable[Any] = self.client.similarity_search(
                    query,
                    k=100,
                    expr="source == 'examples'",
                )
                for d in docs:
                    meta = getattr(d, "metadata", {}) or {}
                    if resources and any(
                        r.uri == meta.get(self.url_field, "")
                        or r.uri == f"milvus://{meta.get(self.id_field, '')}"
                        for r in resources
                    ):
                        continue
                    resources.append(
                        Resource(
                            uri=meta.get(self.url_field, "")
                            or f"milvus://{meta.get(self.id_field, '')}",
                            title=meta.get(self.title_field, "")
                            or meta.get(self.id_field, "Unnamed"),
                            description="Stored Milvus document",
                        )
                    )
                logger.info(
                    "Succeed listed %d resources from Milvus collection: %s",
                    len(resources),
                    self.collection_name,
                )
        except Exception:
            logger.warning(
                "Failed to query Milvus for resources, falling back to local examples."
            )
            return self._list_local_markdown_resources()
        return resources

    def _list_local_markdown_resources(self) -> List[Resource]:
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        examples_path = project_root / self.examples_dir
        if not examples_path.exists():
            return []
        md_files = list(examples_path.glob("*.md"))
        resources: list[Resource] = []
        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                title = self._extract_title_from_markdown(content, md_file.name)
                uri = f"milvus://{self.collection_name}/{md_file.name}"
                resources.append(
                    Resource(
                        uri=uri,
                        title=title,
                        description="Local markdown example (not yet ingested)",
                    )
                )
            except Exception:
                continue
        return resources

    def query_relevant_documents(
        self, query: str, resources: Optional[List[Resource]] = None
    ) -> List[Document]:
        resources = resources or []
        try:
            if not self.client:
                self._connect()
            query_embedding = self._get_embedding(query)
            if self._is_milvus_lite():
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_embedding],
                    anns_field=self.vector_field,
                    param={"metric_type": "IP", "params": {"nprobe": 10}},
                    limit=self.top_k,
                    output_fields=[
                        self.id_field,
                        self.content_field,
                        self.title_field,
                        self.url_field,
                    ],
                )
                documents = {}
                for result_list in search_results:
                    for result in result_list:
                        entity = result.get("entity", {})
                        doc_id = entity.get(self.id_field, "")
                        content = entity.get(self.content_field, "")
                        title = entity.get(self.title_field, "")
                        url = entity.get(self.url_field, "")
                        score = result.get("distance", 0.0)
                        if resources:
                            doc_in_resources = False
                            for resource in resources:
                                if (url and url in resource.uri) or doc_id in resource.uri:
                                    doc_in_resources = True
                                    break
                            if not doc_in_resources:
                                continue
                        if doc_id not in documents:
                            documents[doc_id] = Document(
                                id=doc_id, url=url, title=title, chunks=[]
                            )
                        chunk = Chunk(content=content, similarity=score)
                        documents[doc_id].chunks.append(chunk)
                return list(documents.values())
            else:
                search_results = self.client.similarity_search_with_score(
                    query=query, k=self.top_k
                )
                documents = {}
                for doc, score in search_results:
                    metadata = doc.metadata or {}
                    doc_id = metadata.get(self.id_field, "")
                    title = metadata.get(self.title_field, "")
                    url = metadata.get(self.url_field, "")
                    content = doc.page_content
                    if resources:
                        doc_in_resources = False
                        for resource in resources:
                            if (url and url in resource.uri) or doc_id in resource.uri:
                                doc_in_resources = True
                                break
                        if not doc_in_resources:
                            continue
                    if doc_id not in documents:
                        documents[doc_id] = Document(
                            id=doc_id, url=url, title=title, chunks=[]
                        )
                    chunk = Chunk(content=content, similarity=score)
                    documents[doc_id].chunks.append(chunk)
                return list(documents.values())
        except Exception as e:
            raise RuntimeError(f"Failed to query documents from Milvus: {str(e)}")

    def create_collection(self) -> None:
        if not self.client:
            self._connect()
        else:
            if self._is_milvus_lite():
                self._ensure_collection_exists()

    def load_examples(self, force_reload: bool = False) -> None:
        if not self.client:
            self._connect()
        if force_reload:
            self._clear_example_documents()
        self._load_example_files()

    def _clear_example_documents(self) -> None:
        try:
            if self._is_milvus_lite():
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter="source == 'examples'",
                    output_fields=[self.id_field],
                    limit=10000,
                )
                if results:
                    doc_ids = [result[self.id_field] for result in results]
                    self.client.delete(collection_name=self.collection_name, ids=doc_ids)
                    logger.info("Cleared %d existing example documents", len(doc_ids))
            else:
                logger.info(
                    "Clearing existing examples not supported for LangChain Milvus client"
                )
        except Exception as e:
            logger.warning("Could not clear existing examples: %s", e)

    def get_loaded_examples(self) -> List[Dict[str, str]]:
        try:
            if not self.client:
                self._connect()
            if self._is_milvus_lite():
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter="source == 'examples'",
                    output_fields=[
                        self.id_field,
                        self.title_field,
                        self.url_field,
                        "source",
                        "file",
                    ],
                    limit=1000,
                )
                examples = []
                for result in results:
                    examples.append(
                        {
                            "id": result.get(self.id_field, ""),
                            "title": result.get(self.title_field, ""),
                            "file": result.get("file", ""),
                            "url": result.get(self.url_field, ""),
                        }
                    )
                return examples
            else:
                logger.info(
                    "Getting loaded examples not supported for LangChain Milvus client"
                )
                return []
        except Exception as e:
            logger.error("Error getting loaded examples: %s", e)
            return []

    def close(self) -> None:
        if hasattr(self, "client") and self.client:
            try:
                if self._is_milvus_lite() and hasattr(self.client, "close"):
                    self.client.close()
                self.client = None
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()


class MilvusProvider(MilvusRetriever):
    pass


def load_examples() -> None:
    auto_load_examples = get_bool_env("MILVUS_AUTO_LOAD_EXAMPLES", False)
    rag_provider = get_str_env("RAG_PROVIDER", "")
    if rag_provider == "milvus" and auto_load_examples:
        provider = MilvusProvider()
        provider.load_examples()
