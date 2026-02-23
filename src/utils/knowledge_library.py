"""
Knowledge Library RAG System - Supabase pgvector Implementation.

Uses Supabase pgvector for semantic search.
Benefits: Cloud-based, no local memory issues, scales automatically.
"""

import json
import httpx
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import yaml

from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("knowledge_library")


@dataclass
class RetrievedPassage:
    """Represents a retrieved passage with relevance info."""

    text: str
    source: str
    category: str
    similarity_score: float


class KnowledgeLibrary:
    """
    Manages RAG-based knowledge retrieval using Supabase pgvector.

    Features:
    - Supabase + pgvector for vector storage and search
    - OpenAI text-embedding-3-small (1536 dimensions)
    - Semantic chunking
    - Persistent storage in Supabase
    """

    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536
    TABLE_NAME = "knowledge_chunks"

    def __init__(self, config_path: str = "config/worldview.yaml"):
        self.config = self._load_config(config_path)
        self.library_path = Path(
            self.config.get("knowledge_library", {}).get("library_path", "library/")
        )

        rag_settings = self.config.get("knowledge_library", {}).get("rag_settings", {})
        self.top_k = rag_settings.get("top_k_retrieval", 5)
        self.similarity_threshold = rag_settings.get("similarity_threshold", 0.5)

        self.supabase = None
        self.chunks: List[Dict] = []
        self.initialized = False

        logger.info(
            "KnowledgeLibrary initialized (Supabase pgvector)",
            library_path=str(self.library_path),
        )

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _get_supabase_client(self):
        """Get or create Supabase client."""
        if self.supabase is None:
            from supabase import create_client
            
            supabase_url = settings.api.supabase_url
            supabase_key = settings.api.supabase_anon_key
            
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase URL and anon key must be configured")
            
            self.supabase = create_client(supabase_url, supabase_key)
        return self.supabase

    def _generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API (synchronous)."""
        embeddings = []
        batch_size = 100
        
        headers = {
            "Authorization": f"Bearer {settings.api.openai_api_key}",
            "Content-Type": "application/json",
        }
        
        with httpx.Client(timeout=60.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = client.post(
                    f"{settings.api.openai_base_url}/embeddings",
                    headers=headers,
                    json={
                        "model": self.EMBEDDING_MODEL,
                        "input": batch
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                for embedding_data in data["data"]:
                    embeddings.append(embedding_data["embedding"])
        
        return embeddings

    def _get_query_embedding_sync(self, query: str) -> List[float]:
        """Generate embedding for a query string (synchronous)."""
        headers = {
            "Authorization": f"Bearer {settings.api.openai_api_key}",
            "Content-Type": "application/json",
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{settings.api.openai_base_url}/embeddings",
                headers=headers,
                json={
                    "model": self.EMBEDDING_MODEL,
                    "input": query
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return data["data"][0]["embedding"]

    async def initialize(self) -> bool:
        if self.initialized:
            return True

        client = self._get_supabase_client()
        
        try:
            result = client.table(self.TABLE_NAME).select("id").execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} chunks in Supabase")
                self.chunks = [{"id": row["id"]} for row in result.data]
                self.initialized = True
                return True
            
            logger.info("No existing chunks in Supabase, building index...")
            success = self._build_index_sync()
            self.initialized = success
            return success
            
        except Exception as e:
            logger.warning(f"Failed to load from Supabase, building fresh index: {e}")
            success = self._build_index_sync()
            self.initialized = success
            return success

    def _scan_documents(self) -> List[tuple]:
        """Scan library directory for documents."""
        documents = []

        if not self.library_path.exists():
            logger.warning(f"Library path does not exist: {self.library_path}")
            return []

        categories = self.config.get("knowledge_library", {}).get("categories", [])
        supported_formats = self.config.get("knowledge_library", {}).get(
            "supported_formats", [".pdf", ".md", ".txt"]
        )

        for category in categories:
            category_name = category.get("name", "general")
            category_path = self.library_path / category_name

            if not category_path.exists():
                continue

            for ext in supported_formats:
                for doc_path in category_path.rglob(f"*{ext}"):
                    documents.append(
                        (doc_path, category_name, category.get("relevance_domains", []))
                    )

        return documents

    def _extract_text(self, doc_path: Path) -> str:
        """Extract text from document."""
        try:
            if doc_path.suffix == ".pdf":
                return self._extract_pdf(doc_path)
            elif doc_path.suffix in [".md", ".txt"]:
                with open(doc_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to extract text from {doc_path}: {e}")
        return ""

    def _extract_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using pypdf (lighter than PyMuPDF)."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            logger.error(f"Failed to extract PDF {pdf_path}: {e}")
            return ""

    def _chunk_text(
        self, text: str, chunk_size: int = 512, overlap: int = 128
    ) -> List[str]:
        """Split text into overlapping chunks at sentence boundaries."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_text = (
                    " ".join(current_chunk[-2:]) if len(current_chunk) >= 2 else ""
                )
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = (
                    len(overlap_text) + sentence_len + 1
                    if overlap_text
                    else sentence_len
                )
            else:
                current_chunk.append(sentence)
                current_length += sentence_len + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    def _build_index_sync(self) -> bool:
        """Build index from documents and store in Supabase (synchronous)."""
        try:
            documents = self._scan_documents()

            if not documents:
                logger.warning("No documents found in library")
                return False

            logger.info(f"Found {len(documents)} documents to index")

            all_chunks = []
            for doc_path, category, tags in documents:
                text = self._extract_text(doc_path)
                if not text:
                    continue

                chunks = self._chunk_text(text)
                for i, chunk_text in enumerate(chunks):
                    all_chunks.append(
                        {
                            "text": chunk_text,
                            "source": str(doc_path),
                            "category": category,
                            "tags": json.dumps(tags),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        }
                    )

            if not all_chunks:
                logger.warning("No chunks generated from documents")
                return False

            logger.info(f"Generated {len(all_chunks)} chunks")

            texts = [c["text"] for c in all_chunks]
            logger.info("Generating embeddings via OpenAI...")
            embeddings = self._generate_embeddings_sync(texts)

            client = self._get_supabase_client()
            
            logger.info("Storing chunks and embeddings in Supabase...")
            
            batch_size = 100
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                records = []
                for chunk, embedding in zip(batch, batch_embeddings):
                    record = chunk.copy()
                    record["embedding"] = embedding
                    records.append(record)
                
                client.table(self.TABLE_NAME).insert(records).execute()
                
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")

            self.chunks = all_chunks

            logger.info(f"Index built successfully with {len(all_chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Failed to build index: {e}", exc_info=True)
            return False

    async def retrieve_relevant_passages(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> List[RetrievedPassage]:
        if not self.initialized:
            logger.warning("Knowledge library not initialized")
            return []

        top_k = top_k or self.top_k

        try:
            query_embedding = self._get_query_embedding_sync(query)
            
            client = self._get_supabase_client()
            
            search_k = top_k * 3
            
            result = client.rpc("match_knowledge_chunks", {
                "query_embedding": query_embedding,
                "match_threshold": self.similarity_threshold,
                "match_count": search_k
            }).execute()
            
            passages = []
            for row in result.data:
                if category_filter and row.get("category") != category_filter:
                    continue
                
                similarity = row.get("similarity", 0)
                
                if similarity < self.similarity_threshold:
                    continue

                passages.append(
                    RetrievedPassage(
                        text=row["text"][:1000],
                        source=row["source"],
                        category=row["category"],
                        similarity_score=float(similarity),
                    )
                )

                if len(passages) >= top_k:
                    break

            logger.debug(f"Retrieved {len(passages)} passages for: {query[:50]}...")
            return passages

        except Exception as e:
            logger.error(f"Failed to retrieve passages: {e}", exc_info=True)
            return []

    async def refresh_index(self) -> bool:
        """Rebuild index from scratch."""
        logger.info("Refreshing knowledge library index...")
        
        try:
            client = self._get_supabase_client()
            client.table(self.TABLE_NAME).delete().neq("id", 0).execute()
        except Exception as e:
            logger.warning(f"Failed to clear table: {e}")
        
        self.chunks = []
        self.initialized = False
        return await self.initialize()

    def get_library_stats(self) -> Dict:
        return {
            "initialized": self.initialized,
            "num_chunks": len(self.chunks),
            "library_path": str(self.library_path),
            "storage": "supabase",
        }


FAISS_AVAILABLE = False
OPENAI_AVAILABLE = True
PYMUPDF_AVAILABLE = True


@dataclass
class DocumentChunk:
    """Backward compatibility - represents a chunk of a document."""

    text: str
    source: str
    category: str
    tags: List[str]
    chunk_index: int
    total_chunks: int
    embedding: Optional[List[float]] = None


_knowledge_library: Optional[KnowledgeLibrary] = None


async def get_knowledge_library() -> KnowledgeLibrary:
    global _knowledge_library
    if _knowledge_library is None:
        _knowledge_library = KnowledgeLibrary()
        await _knowledge_library.initialize()
    return _knowledge_library
