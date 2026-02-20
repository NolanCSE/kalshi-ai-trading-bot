"""
Knowledge Library RAG System for Knowledge Researcher Agent.

Manages document ingestion, vector storage, and retrieval for worldview-based analysis.
Uses FAISS for local vector storage and sentence-transformers for local embeddings (zero API cost).

Key improvements:
- Local embeddings via sentence-transformers (no API calls, faster, zero cost)
- Structure-aware chunking for books (preserves chapters/sections)
- Query result caching (avoids repeated searches)
- Hierarchical chunking strategy for long documents
"""

import os
import pickle
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import time

import yaml
import numpy as np

from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("knowledge_library")

# Optional imports - handle gracefully if not installed
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Knowledge library will use fallback mode.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. PDF processing unavailable.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Will use OpenAI fallback.")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain-text-splitters not installed. Using basic chunking.")


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    text: str
    source: str  # File path
    category: str
    tags: List[str]
    chunk_index: int
    total_chunks: int
    embedding: Optional[List[float]] = None
    # Structure-aware metadata
    section_title: Optional[str] = None  # Chapter/section heading
    section_level: int = 0  # Hierarchy level (0=book, 1=chapter, 2=section)
    word_count: int = 0


@dataclass
class RetrievedPassage:
    """Represents a retrieved passage with relevance info."""
    text: str
    source: str
    category: str
    similarity_score: float
    context_before: str = ""
    context_after: str = ""
    section_title: Optional[str] = None  # For structure-aware context


@dataclass
class CacheEntry:
    """Query cache entry."""
    results: List[RetrievedPassage]
    timestamp: datetime
    query_hash: str


class QueryCache:
    """Simple in-memory cache for query results."""
    
    def __init__(self, ttl_minutes: int = 60, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_size = max_size
        self.access_times: Dict[str, datetime] = {}
    
    def _hash_query(self, query: str, category_filter: Optional[str]) -> str:
        """Create hash key for query."""
        key = f"{query}:{category_filter or 'all'}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, query: str, category_filter: Optional[str]) -> Optional[List[RetrievedPassage]]:
        """Get cached results if valid."""
        key = self._hash_query(query, category_filter)
        entry = self.cache.get(key)
        
        if entry is None:
            return None
        
        # Check TTL
        if datetime.now() - entry.timestamp > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            return None
        
        # Update access time for LRU
        self.access_times[key] = datetime.now()
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return entry.results
    
    def set(self, query: str, category_filter: Optional[str], results: List[RetrievedPassage]):
        """Cache query results."""
        key = self._hash_query(query, category_filter)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = CacheEntry(
            results=results,
            timestamp=datetime.now(),
            query_hash=key
        )
        self.access_times[key] = datetime.now()
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.access_times.clear()
    
    def invalidate(self):
        """Invalidate cache (call when index is rebuilt)."""
        self.clear()


class LocalEmbedder:
    """Local embedding model using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # all-MiniLM-L6-v2 output dimension
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available")
            return
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            start = time.time()
            self.model = SentenceTransformer(self.model_name)
            elapsed = time.time() - start
            logger.info(f"Embedding model loaded in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        # Normalize embeddings for cosine similarity
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embeddings.astype("float32")
    
    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query."""
        embeddings = self.encode([text])
        return embeddings[0]


class StructureAwareChunker:
    """Chunks documents while preserving structure (chapters, sections)."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Patterns for detecting structure
        self.chapter_patterns = [
            r'^Chapter\s+\d+',  # Chapter 1, Chapter 2
            r'^CHAPTER\s+\d+',
            r'^\d+\.\s+',  # 1. Introduction
            r'^Part\s+\d+',  # Part I, Part 1
            r'^Section\s+\d+',
        ]
        
        self.heading_patterns = [
            r'^#{1,6}\s+',  # Markdown headings
            r'^\d+\.\d+\s+',  # 1.1, 1.2 subsections
        ]
    
    def _detect_structure(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Detect document structure and return sections.
        Returns: List of (section_title, section_content, level)
        """
        lines = text.split('\n')
        sections = []
        current_title = "Introduction"
        current_content = []
        current_level = 1
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for chapter/section headers
            is_header = False
            new_level = current_level
            
            for pattern in self.chapter_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_header = True
                    new_level = 1
                    break
            
            if not is_header:
                for pattern in self.heading_patterns:
                    if re.match(pattern, line_stripped):
                        is_header = True
                        # Count # for markdown level
                        if line_stripped.startswith('#'):
                            new_level = len(line_stripped) - len(line_stripped.lstrip('#'))
                        else:
                            new_level = 2
                        break
            
            if is_header:
                # Save previous section
                if current_content:
                    sections.append((
                        current_title,
                        '\n'.join(current_content),
                        current_level
                    ))
                # Start new section
                current_title = line_stripped
                current_content = []
                current_level = new_level
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            sections.append((
                current_title,
                '\n'.join(current_content),
                current_level
            ))
        
        return sections
    
    def chunk_document(self, text: str, source: str) -> List[Tuple[str, str, int]]:
        """
        Chunk document preserving structure.
        Returns: List of (chunk_text, section_title, level)
        """
        # Detect structure
        sections = self._detect_structure(text)
        
        if not sections:
            # No structure detected, treat as single section
            sections = [("Content", text, 0)]
        
        chunks = []
        
        for section_title, section_content, level in sections:
            section_word_count = len(section_content.split())
            
            if section_word_count <= self.chunk_size:
                # Small section, keep as single chunk
                chunks.append((section_content, section_title, level))
            else:
                # Large section, need to split
                if LANGCHAIN_AVAILABLE:
                    # Use langchain for smart splitting
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
                        length_function=lambda x: len(x.split()),  # Word count
                    )
                    sub_chunks = splitter.split_text(section_content)
                else:
                    # Fallback to simple splitting
                    sub_chunks = self._simple_split(section_content)
                
                for chunk in sub_chunks:
                    chunks.append((chunk, section_title, level))
        
        return chunks
    
    def _simple_split(self, text: str) -> List[str]:
        """Simple word-based splitting with overlap."""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            # Move with overlap
            start = end - self.chunk_overlap
            if start >= end:
                start = end
        
        return chunks


class KnowledgeLibrary:
    """
    Manages RAG-based knowledge retrieval for the Knowledge Researcher.
    
    Features:
    - Local embeddings via sentence-transformers (zero API cost)
    - Structure-aware chunking for books (preserves chapters/sections)
    - Query result caching
    - Document ingestion (PDF, Markdown, TXT)
    - FAISS vector storage
    - Semantic search retrieval
    """
    
    def __init__(self, config_path: str = "config/worldview.yaml"):
        """Initialize knowledge library."""
        self.config = self._load_config(config_path)
        self.library_path = Path(self.config.get("knowledge_library", {}).get("library_path", "library/"))
        self.cache_dir = Path(".cache/knowledge_library")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # RAG settings
        rag_settings = self.config.get("knowledge_library", {}).get("rag_settings", {})
        self.chunk_size = rag_settings.get("chunk_size", 512)
        self.chunk_overlap = rag_settings.get("chunk_overlap", 128)
        self.top_k = rag_settings.get("top_k_retrieval", 5)
        self.similarity_threshold = rag_settings.get("similarity_threshold", 0.70)
        
        # Embedding model selection
        embedding_config = rag_settings.get("embedding_model", "local")
        if embedding_config == "local" and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.use_local_embeddings = True
            self.embedder = LocalEmbedder("all-MiniLM-L6-v2")
            self.embedding_dim = self.embedder.dimension
            logger.info("Using local embeddings (sentence-transformers)")
        else:
            self.use_local_embeddings = False
            self.embedder = None
            self.embedding_dim = 1536  # OpenAI text-embedding-3-small
            logger.info("Using OpenAI embeddings (API calls required)")
        
        # Initialize chunker
        self.chunker = StructureAwareChunker(self.chunk_size, self.chunk_overlap)
        
        # Initialize query cache
        cache_config = rag_settings.get("query_cache", {})
        self.query_cache = QueryCache(
            ttl_minutes=cache_config.get("ttl_minutes", 60),
            max_size=cache_config.get("max_size", 1000)
        )
        
        # State
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.initialized = False
        
        logger.info(
            "KnowledgeLibrary initialized",
            library_path=str(self.library_path),
            chunk_size=self.chunk_size,
            use_local_embeddings=self.use_local_embeddings,
            faiss_available=FAISS_AVAILABLE,
        )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load worldview configuration."""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    async def initialize(self) -> bool:
        """
        Initialize the knowledge library.
        Loads existing index or builds from scratch.
        """
        if self.initialized:
            return True
        
        if not FAISS_AVAILABLE:
            logger.warning("Knowledge library operating in fallback mode - FAISS not available")
            return False
        
        # Try to load cached index
        if await self._load_cached_index():
            logger.info("Loaded cached knowledge library index")
            self.initialized = True
            return True
        
        # Build index from documents
        logger.info("Building knowledge library index from documents...")
        success = await self._build_index()
        self.initialized = success
        return success
    
    async def _build_index(self) -> bool:
        """Build FAISS index from documents in library."""
        try:
            # Scan for documents
            documents = await self._scan_documents()
            if not documents:
                logger.warning("No documents found in library")
                return False
            
            logger.info(f"Found {len(documents)} documents to index")
            
            # Process documents into chunks
            all_chunks = []
            for doc_path, category, tags in documents:
                chunks = await self._process_document(doc_path, category, tags)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                logger.warning("No chunks generated from documents")
                return False
            
            logger.info(f"Generated {len(all_chunks)} chunks")
            
            # Generate embeddings
            await self._generate_embeddings(all_chunks)
            
            # Build FAISS index
            await self._build_faiss_index(all_chunks)
            
            # Save cache
            await self._save_cached_index()
            
            # Invalidate query cache since index changed
            self.query_cache.invalidate()
            
            logger.info("Knowledge library index built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}", exc_info=True)
            return False
    
    async def _scan_documents(self) -> List[Tuple[Path, str, List[str]]]:
        """Scan library directory for documents."""
        documents = []
        
        if not self.library_path.exists():
            logger.warning(f"Library path does not exist: {self.library_path}")
            return []
        
        categories = self.config.get("knowledge_library", {}).get("categories", [])
        supported_formats = self.config.get("knowledge_library", {}).get("supported_formats", [".pdf", ".md", ".txt"])
        
        for category in categories:
            category_name = category.get("name", "general")
            category_path = self.library_path / category_name
            
            if not category_path.exists():
                continue
            
            for ext in supported_formats:
                for doc_path in category_path.rglob(f"*{ext}"):
                    documents.append((
                        doc_path,
                        category_name,
                        category.get("relevance_domains", [])
                    ))
        
        return documents
    
    async def _process_document(
        self,
        doc_path: Path,
        category: str,
        tags: List[str]
    ) -> List[DocumentChunk]:
        """Process a document into chunks using structure-aware chunking."""
        try:
            # Extract text based on file type
            if doc_path.suffix == ".pdf":
                text = await self._extract_pdf_text(doc_path)
            elif doc_path.suffix in [".md", ".txt"]:
                text = await self._extract_text_file(doc_path)
            else:
                return []
            
            if not text:
                return []
            
            # Use structure-aware chunking
            chunk_tuples = self.chunker.chunk_document(text, str(doc_path))
            
            # Create DocumentChunk objects
            doc_chunks = []
            for i, (chunk_text, section_title, level) in enumerate(chunk_tuples):
                doc_chunks.append(DocumentChunk(
                    text=chunk_text,
                    source=str(doc_path),
                    category=category,
                    tags=tags,
                    chunk_index=i,
                    total_chunks=len(chunk_tuples),
                    section_title=section_title,
                    section_level=level,
                    word_count=len(chunk_text.split())
                ))
            
            logger.debug(f"Processed {doc_path.name} into {len(doc_chunks)} chunks")
            return doc_chunks
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_path}: {e}")
            return []
    
    async def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available, cannot process PDF")
            return ""
        
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Failed to extract PDF {pdf_path}: {e}")
            return ""
    
    async def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from markdown or text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return ""
    
    async def _generate_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings for document chunks."""
        if self.use_local_embeddings and self.embedder:
            await self._generate_local_embeddings(chunks)
        else:
            await self._generate_openai_embeddings(chunks)
    
    async def _generate_local_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings using local sentence-transformers model."""
        if self.embedder is None:
            logger.error("Local embedder not available")
            return
        
        try:
            texts = [chunk.text for chunk in chunks]
            
            # Run in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,  # Default executor
                lambda: self.embedder.encode(texts, batch_size=32)
            )
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()
            
            logger.info(f"Generated {len(chunks)} local embeddings")
            
        except Exception as e:
            logger.error(f"Failed to generate local embeddings: {e}")
    
    async def _generate_openai_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings using OpenAI API (fallback)."""
        from src.config.settings import settings
        
        api_key = settings.api.openai_api_key
        if not api_key:
            logger.error("OpenAI API key not found")
            return
        
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)
            
            # Process in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                texts = [chunk.text for chunk in batch]
                
                response = await client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                )
                
                for chunk, embedding_data in zip(batch, response.data):
                    chunk.embedding = embedding_data.embedding
                
                logger.debug(f"Generated OpenAI embeddings for batch {i//batch_size + 1}")
                
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embeddings: {e}")
    
    async def _build_faiss_index(self, chunks: List[DocumentChunk]) -> None:
        """Build FAISS index from chunks with embeddings."""
        # Filter chunks with embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        
        if not valid_chunks:
            logger.error("No chunks with embeddings to index")
            return
        
        # Convert to numpy array
        embeddings = np.array([c.embedding for c in valid_chunks]).astype("float32")
        
        # Create FAISS index
        dimension = len(valid_chunks[0].embedding)
        
        # Use IndexFlatIP for small libraries, HNSW for large ones
        if len(valid_chunks) < 1000:
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"Using IndexFlatIP for {len(valid_chunks)} chunks")
        else:
            # HNSW for faster search on large libraries
            self.index = faiss.IndexHNSWFlat(dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
            logger.info(f"Using IndexHNSWFlat for {len(valid_chunks)} chunks")
        
        # Add embeddings to index
        self.index.add(embeddings)
        self.chunks = valid_chunks
        
        logger.info(f"Built FAISS index with {len(valid_chunks)} chunks, dimension {dimension}")
    
    async def retrieve_relevant_passages(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None
    ) -> List[RetrievedPassage]:
        """
        Retrieve relevant passages for a query with caching.
        
        Args:
            query: Search query
            top_k: Number of results (defaults to config setting)
            category_filter: Optional category to filter by
            
        Returns:
            List of RetrievedPassage objects
        """
        if not self.initialized or self.index is None:
            logger.warning("Knowledge library not initialized")
            return []
        
        top_k = top_k or self.top_k
        
        # Check cache first
        cached_results = self.query_cache.get(query, category_filter)
        if cached_results is not None:
            return cached_results
        
        try:
            # Generate query embedding
            query_embedding = await self._get_query_embedding(query)
            if query_embedding is None:
                return []
            
            # Search FAISS index
            query_vector = np.array([query_embedding]).astype("float32")
            
            # Search with extra results for filtering
            search_k = top_k * 3 if category_filter else top_k
            scores, indices = self.index.search(query_vector, search_k)
            
            # Build results
            passages = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self.chunks):
                    continue
                
                chunk = self.chunks[idx]
                
                # Apply category filter if specified
                if category_filter and chunk.category != category_filter:
                    continue
                
                # Apply similarity threshold
                if score < self.similarity_threshold:
                    continue
                
                passages.append(RetrievedPassage(
                    text=chunk.text,
                    source=chunk.source,
                    category=chunk.category,
                    similarity_score=float(score),
                    section_title=chunk.section_title
                ))
                
                if len(passages) >= top_k:
                    break
            
            logger.debug(f"Retrieved {len(passages)} passages for query: {query[:50]}...")
            
            # Cache results
            self.query_cache.set(query, category_filter, passages)
            
            return passages
            
        except Exception as e:
            logger.error(f"Failed to retrieve passages: {e}", exc_info=True)
            return []
    
    async def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for query text."""
        try:
            if self.use_local_embeddings and self.embedder:
                # Use local embedder
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None,
                    lambda: self.embedder.encode_query(query)
                )
                return embedding
            else:
                # Fallback to OpenAI
                return await self._get_openai_query_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None
    
    async def _get_openai_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API (fallback)."""
        from src.config.settings import settings
        
        api_key = settings.api.openai_api_key
        if not api_key:
            logger.error("OpenAI API key not found")
            return None
        
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=[query]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate OpenAI query embedding: {e}")
            return None
    
    async def _load_cached_index(self) -> bool:
        """Load cached FAISS index if available and valid."""
        try:
            index_path = self.cache_dir / "faiss_index.bin"
            chunks_path = self.cache_dir / "chunks.pkl"
            metadata_path = self.cache_dir / "metadata.pkl"
            
            if not index_path.exists() or not chunks_path.exists():
                return False
            
            # Check if library has changed
            if not await self._is_cache_valid():
                logger.info("Cache is stale, rebuilding index")
                return False
            
            # Load index
            self.index = faiss.read_index(str(index_path))
            
            # Load chunks
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            
            logger.info(f"Loaded cached index with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cached index: {e}")
            return False
    
    async def _save_cached_index(self) -> None:
        """Save FAISS index and chunks to cache."""
        try:
            index_path = self.cache_dir / "faiss_index.bin"
            chunks_path = self.cache_dir / "chunks.pkl"
            metadata_path = self.cache_dir / "metadata.pkl"
            
            # Save index
            faiss.write_index(self.index, str(index_path))
            
            # Save chunks
            with open(chunks_path, "wb") as f:
                pickle.dump(self.chunks, f)
            
            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "num_chunks": len(self.chunks),
                "library_hash": await self._compute_library_hash(),
                "use_local_embeddings": self.use_local_embeddings,
            }
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            
            logger.info("Saved index cache")
            
        except Exception as e:
            logger.error(f"Failed to save cached index: {e}")
    
    async def _is_cache_valid(self) -> bool:
        """Check if cached index is still valid."""
        try:
            metadata_path = self.cache_dir / "metadata.pkl"
            if not metadata_path.exists():
                return False
            
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            current_hash = await self._compute_library_hash()
            cached_hash = metadata.get("library_hash")
            
            return current_hash == cached_hash
            
        except Exception as e:
            logger.error(f"Failed to check cache validity: {e}")
            return False
    
    async def _compute_library_hash(self) -> str:
        """Compute hash of library contents for cache validation."""
        try:
            documents = await self._scan_documents()
            if not documents:
                return ""
            
            # Hash file paths and modification times
            hash_content = []
            for doc_path, category, tags in sorted(documents, key=lambda x: str(x[0])):
                stat = doc_path.stat()
                hash_content.append(f"{doc_path}:{stat.st_mtime}:{stat.st_size}")
            
            return hashlib.md5("\n".join(hash_content).encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to compute library hash: {e}")
            return ""


# Singleton instance
_knowledge_library: Optional[KnowledgeLibrary] = None


async def get_knowledge_library() -> KnowledgeLibrary:
    """Get or create the singleton knowledge library instance."""
    global _knowledge_library
    if _knowledge_library is None:
        _knowledge_library = KnowledgeLibrary()
        await _knowledge_library.initialize()
    return _knowledge_library


async def reset_knowledge_library():
    """Reset the singleton instance (useful for testing)."""
    global _knowledge_library
    _knowledge_library = None
