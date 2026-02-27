"""
Knowledge Library RAG System - Supabase pgvector Implementation.

Uses Supabase pgvector for semantic search with streaming document processing
to avoid memory issues on local machines.
"""

import json
import gc
import hashlib
import httpx
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

import yaml

from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.utils.encryption import TextEncryptor

logger = get_trading_logger("knowledge_library")


@dataclass
class RetrievedPassage:
    """Represents a retrieved passage with relevance info."""

    text: str
    source: str
    category: str
    similarity_score: float
    page_number: Optional[int] = None


class KnowledgeLibrary:
    """
    Manages RAG-based knowledge retrieval using Supabase pgvector.

    Features:
    - Supabase + pgvector for vector storage and search
    - OpenAI text-embedding-3-small (1536 dimensions)
    - Batch processing: embed + insert every 100 chunks (saves progress)
    - Incremental updates: only process new documents
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
        self.chunk_count = 0
        self.initialized = False
        self._existing_sources: Set[str] = set()
        self._encryptor = TextEncryptor()

        logger.info(
            "KnowledgeLibrary initialized (Supabase pgvector - batch mode)",
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
        if not texts:
            return []
            
        embeddings = []
        batch_size = 25
        
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
                
                logger.debug(f"Generated embeddings {i+1}-{i+len(batch)}/{len(texts)}")
        
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

    def _extract_pdf_page_by_page(self, pdf_path: Path):
        """Extract text from PDF one page at a time to minimize memory."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            logger.info(f"Processing PDF with {total_pages} pages: {pdf_path.name}")
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    yield text
                    
                if (page_num + 1) % 50 == 0:
                    logger.info(f"Extracted {page_num + 1}/{total_pages} pages")
                    
        except Exception as e:
            logger.error(f"Failed to extract PDF {pdf_path}: {e}")

    def _extract_text_file(self, doc_path: Path) -> str:
        """Extract text from markdown or text file."""
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to extract text from {doc_path}: {e}")
            return ""

    def _chunk_text(
        self, text: str, chunk_size: int = 512, overlap: int = 128
    ) -> List[str]:
        """Split text into overlapping chunks at sentence boundaries.
        
        Uses proper character-based overlap: takes only the last N characters
        from the previous chunk, not full sentences.
        """
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Take only the last N characters as overlap (proper overlap)
                overlap_text = (" ".join(current_chunk))[-overlap:] if overlap > 0 else ""
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

    def _load_existing_sources(self) -> Set[str]:
        """Load all existing source paths from Supabase (by hash for comparison)."""
        client = self._get_supabase_client()
        result = client.table(self.TABLE_NAME).select("source_hash").execute()
        sources = {str(row["source_hash"]) for row in result.data}
        return sources

    def _get_max_chunk_index(self, source_path: str) -> int:
        """Get the highest chunk_index already stored for a source."""
        source_hash = hashlib.sha256(source_path.encode()).hexdigest()[:16]
        client = self._get_supabase_client()
        result = client.table(self.TABLE_NAME).select("chunk_index").eq("source_hash", source_hash).order("chunk_index", desc=True).limit(1).execute()
        if result.data:
            return int(result.data[0]["chunk_index"])
        return -1
    
    def _get_total_chunk_count(self) -> int:
        """Get total number of chunks in database."""
        client = self._get_supabase_client()
        result = client.table(self.TABLE_NAME).select("id", count="exact").execute()
        return result.count if result.count else 0

    def _delete_chunks_by_source(self, source_path: str) -> int:
        """Delete all chunks for a specific source file."""
        client = self._get_supabase_client()
        result = client.table(self.TABLE_NAME).delete().eq("source", source_path).execute()
        return len(result.data) if result.data else 0

    def _delete_chunks_by_hash(self, source_hash: str) -> int:
        """Delete all chunks for a specific source by hash."""
        client = self._get_supabase_client()
        result = client.table(self.TABLE_NAME).delete().eq("source_hash", source_hash).execute()
        return len(result.data) if result.data else 0

    def _process_single_document(
        self, 
        doc_path: Path, 
        category: str, 
        tags: List[str],
        client,
        delete_existing: bool = False
    ) -> int:
        """Process a single document: embed + insert in batches of 100.
        
        Each batch is inserted immediately to save progress if interrupted.
        Skips already-embedded chunks if interrupted and resumed.
        Tracks page numbers for citations.
        """
        
        source_path = str(doc_path)
        
        if delete_existing:
            deleted = self._delete_chunks_by_source(source_path)
            if deleted > 0:
                logger.info(f"Removed {deleted} old chunks for {doc_path.name}")
        
        # Check existing chunks and skip ahead if partially indexed
        existing_max_idx = self._get_max_chunk_index(source_path)
        start_chunk_idx = existing_max_idx + 1
        logger.info(f"Existing max chunk index: {existing_max_idx}, starting from: {start_chunk_idx}")
        
        # Extract text with page tracking
        if doc_path.suffix == ".pdf":
            all_chunks_with_pages = []
            
            for page_num, page_text in enumerate(self._extract_pdf_page_by_page(doc_path), start=1):
                if not page_text.strip():
                    continue
                page_chunks = self._chunk_text(page_text)
                for chunk in page_chunks:
                    all_chunks_with_pages.append((chunk, page_num))
            
            if not all_chunks_with_pages:
                return 0
            
            total_chunks = len(all_chunks_with_pages)
            logger.info(f"Generated {total_chunks} chunks from {doc_path.name}")
            
            # Separate chunks and pages for processing
            all_chunks = [c[0] for c in all_chunks_with_pages]
            all_pages = [c[1] for c in all_chunks_with_pages]
            
        else:
            text = self._extract_text_file(doc_path)
            if not text.strip():
                return 0
            all_chunks = self._chunk_text(text)
            all_pages = [None] * len(all_chunks)
            total_chunks = len(all_chunks)
        
        # Skip already-embedded chunks
        if start_chunk_idx >= total_chunks:
            logger.info(f"Document already fully indexed ({total_chunks} chunks), skipping")
            return 0
        
        # Process in batches: embed + insert each batch IMMEDIATELY
        # Start from the first un-embedded chunk
        batch_size = 100
        total_inserted = 0
        
        for batch_start in range(start_chunk_idx, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_chunks = all_chunks[batch_start:batch_end]
            batch_pages = all_pages[batch_start:batch_end]
            
            # Generate embeddings for this batch
            batch_embeddings = self._generate_embeddings_sync(batch_chunks)
            
            # Create records and insert immediately
            records = []
            for i, (chunk_text, embedding, page_num) in enumerate(zip(batch_chunks, batch_embeddings, batch_pages)):
                records.append({
                    "text": self._encryptor.encrypt(chunk_text),
                    "source": self._encryptor.encrypt(source_path),
                    "source_hash": hashlib.sha256(source_path.encode()).hexdigest()[:16],
                    "category": self._encryptor.encrypt(category),
                    "tags": json.dumps(tags),
                    "chunk_index": batch_start + i,
                    "total_chunks": total_chunks,
                    "page_number": self._encryptor.encrypt(str(page_num)) if page_num else None,
                    "embedding": embedding,
                })
            
            # Insert NOW (saves progress!)
            client.table(self.TABLE_NAME).insert(records).execute()
            total_inserted += len(records)
            
            batches_total = (total_chunks + batch_size - 1) // batch_size
            batches_done = batch_start // batch_size + 1
            logger.info(f"Batch {batches_done}/{batches_total}: {total_inserted}/{total_chunks} chunks saved")
            
            del batch_embeddings
            del records
            gc.collect()
        
        del all_chunks
        gc.collect()
        
        logger.info(f"Completed {doc_path.name}: {total_inserted} chunks")
        return total_inserted

    def _build_index_incremental(self) -> bool:
        """Build index incrementally: only process new/partial documents."""
        try:
            documents = self._scan_documents()

            if not documents:
                logger.warning("No documents found in library")
                return False

            logger.info(f"Found {len(documents)} documents to check")

            # Get existing sources from DB
            self._existing_sources = self._load_existing_sources()
            current_sources = {str(doc[0]) for doc in documents}
            
            # Find documents that need processing (new or partially indexed)
            docs_to_process = []
            
            for doc in documents:
                doc_path_str = str(doc[0])
                doc_hash = hashlib.sha256(doc_path_str.encode()).hexdigest()[:16]
                if doc_hash not in self._existing_sources:
                    # New document
                    docs_to_process.append(doc)
                else:
                    # Check if fully indexed by looking at max chunk_index
                    max_idx = self._get_max_chunk_index(doc_path_str)
                    # Total chunks expected (will need to re-extract to know exact)
                    # For now, assume if it exists it might be partial - we'll check inside
                    docs_to_process.append(doc)
            
            # Find documents to remove (in DB but not on disk)
            sources_to_remove = self._existing_sources - current_sources

            # Remove deleted documents
            for source_hash in sources_to_remove:
                deleted = self._delete_chunks_by_hash(source_hash)
                logger.info(f"Removed {deleted} chunks for deleted file: {source_hash}")

            if not docs_to_process:
                logger.info("No documents to process")
                self.chunk_count = self._get_total_chunk_count()
                return True

            logger.info(f"Need to process {len(docs_to_process)} documents")

            client = self._get_supabase_client()
            total_chunks = 0

            for doc_idx, (doc_path, category, tags) in enumerate(docs_to_process):
                logger.info(f"Processing document {doc_idx + 1}/{len(docs_to_process)}: {doc_path.name}")
                
                try:
                    chunks_count = self._process_single_document(
                        doc_path, category, tags, client, delete_existing=False
                    )
                    total_chunks += chunks_count
                    logger.info(f"Completed {doc_path.name}: {chunks_count} chunks")
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Failed to process {doc_path}: {e}")
                    continue

            # Update total chunk count
            result = client.table(self.TABLE_NAME).select("id").execute()
            self.chunk_count = len(result.data)
            
            logger.info(f"Index updated: added {total_chunks} chunks, total now: {self.chunk_count}")
            return True

        except Exception as e:
            logger.error(f"Failed to build index: {e}", exc_info=True)
            return False

    async def initialize(self) -> bool:
        if self.initialized:
            return True

        client = self._get_supabase_client()
        
        try:
            result = client.table(self.TABLE_NAME).select("id").execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} chunks in Supabase (using incremental mode)")
                self.chunk_count = len(result.data)
                self._existing_sources = self._load_existing_sources()
                self.initialized = True
                
                # Check for new/removed docs
                self._build_index_incremental()
                return True
            
            logger.info("No existing chunks in Supabase, building index...")
            success = self._build_index_incremental()
            self.initialized = success
            return success
            
        except Exception as e:
            logger.warning(f"Failed to load from Supabase, building fresh index: {e}")
            success = self._build_index_incremental()
            self.initialized = success
            return success

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

                page_num = row.get("page_number")
                try:
                    page_num = int(self._encryptor.decrypt(page_num)) if page_num else None
                except Exception:
                    page_num = None

                passages.append(
                    RetrievedPassage(
                        text=self._encryptor.decrypt(row["text"])[:1000],
                        source=self._encryptor.decrypt(row["source"]),
                        category=self._encryptor.decrypt(row["category"]),
                        similarity_score=float(similarity),
                        page_number=page_num,
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
        logger.info("Refreshing knowledge library index (full rebuild)...")
        
        try:
            client = self._get_supabase_client()
            client.table(self.TABLE_NAME).delete().neq("id", 0).execute()
        except Exception as e:
            logger.warning(f"Failed to clear table: {e}")
        
        self.chunk_count = 0
        self._existing_sources = set()
        self.initialized = False
        return await self.initialize()

    def get_library_stats(self) -> Dict:
        return {
            "initialized": self.initialized,
            "num_chunks": self.chunk_count,
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
