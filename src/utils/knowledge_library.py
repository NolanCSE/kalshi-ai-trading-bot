"""
Knowledge Library RAG System for Ideology Agent.

Manages document ingestion, vector storage, and retrieval for worldview-based analysis.
Uses FAISS for local vector storage and OpenAI embeddings.
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio

import yaml

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
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Knowledge library will use fallback mode.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. PDF processing unavailable.")


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


@dataclass
class RetrievedPassage:
    """Represents a retrieved passage with relevance info."""
    text: str
    source: str
    category: str
    similarity_score: float
    context_before: str = ""
    context_after: str = ""


class KnowledgeLibrary:
    """
    Manages RAG-based knowledge retrieval for the Ideology Agent.
    
    Features:
    - Document ingestion (PDF, Markdown, TXT)
    - Chunking with overlap
    - FAISS vector storage
    - Semantic search retrieval
    - Cache management
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
        self.similarity_threshold = rag_settings.get("similarity_threshold", 0.75)
        self.embedding_model = rag_settings.get("embedding_model", "text-embedding-3-small")
        
        # State
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.initialized = False
        
        logger.info(
            "KnowledgeLibrary initialized",
            library_path=str(self.library_path),
            chunk_size=self.chunk_size,
            faiss_available=FAISS_AVAILABLE,
            openai_available=OPENAI_AVAILABLE,
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
            
        if not FAISS_AVAILABLE or not OPENAI_AVAILABLE:
            logger.warning("Knowledge library operating in fallback mode - missing dependencies")
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
        """Process a document into chunks."""
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
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            # Create DocumentChunk objects
            doc_chunks = []
            for i, chunk_text in enumerate(chunks):
                doc_chunks.append(DocumentChunk(
                    text=chunk_text,
                    source=str(doc_path),
                    category=category,
                    tags=tags,
                    chunk_index=i,
                    total_chunks=len(chunks)
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
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        # Simple chunking by character count with overlap
        # More sophisticated chunking could use sentence boundaries
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence ending punctuation
                for i in range(min(100, end - start)):
                    if text[end - i - 1] in ".!?":
                        end = end - i
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start forward with overlap
            start = end - self.chunk_overlap
            if start >= end:
                start = end
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """Generate embeddings for document chunks using OpenAI."""
        from src.config.settings import settings
        
        api_key = settings.api.openai_api_key
        if not api_key:
            logger.error("OpenAI API key not found")
            return
        
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Process in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [chunk.text for chunk in batch]
            
            try:
                response = await client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                
                for chunk, embedding_data in zip(batch, response.data):
                    chunk.embedding = embedding_data.embedding
                
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
    
    async def _build_faiss_index(self, chunks: List[DocumentChunk]) -> None:
        """Build FAISS index from chunks with embeddings."""
        # Filter chunks with embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        
        if not valid_chunks:
            logger.error("No chunks with embeddings to index")
            return
        
        # Convert to numpy array
        import numpy as np
        embeddings = np.array([c.embedding for c in valid_chunks]).astype("float32")
        
        # Create FAISS index
        dimension = len(valid_chunks[0].embedding)
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
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
        Retrieve relevant passages for a query.
        
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
        
        try:
            # Generate query embedding
            query_embedding = await self._get_query_embedding(query)
            if query_embedding is None:
                return []
            
            # Search FAISS index
            import numpy as np
            query_vector = np.array([query_embedding]).astype("float32")
            faiss.normalize_L2(query_vector)
            
            scores, indices = self.index.search(query_vector, top_k * 2)  # Get extra for filtering
            
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
                    similarity_score=float(score)
                ))
                
                if len(passages) >= top_k:
                    break
            
            logger.debug(f"Retrieved {len(passages)} passages for query: {query[:50]}...")
            return passages
            
        except Exception as e:
            logger.error(f"Failed to retrieve passages: {e}", exc_info=True)
            return []
    
    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for query text."""
        from src.config.settings import settings
        
        api_key = settings.api.openai_api_key
        if not api_key:
            logger.error("OpenAI API key not found")
            return None
        
        try:
            client = openai.AsyncOpenAI(api_key=api_key)
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None
    
    async def _load_cached_index(self) -> bool:
        """Load cached FAISS index if available and valid."""
        index_path = self.cache_dir / "faiss_index.bin"
        chunks_path = self.cache_dir / "chunks.pkl"
        metadata_path = self.cache_dir / "metadata.pkl"
        
        if not index_path.exists() or not chunks_path.exists():
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load chunks
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            
            logger.info(f"Loaded cached index with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cached index: {e}")
            return False
    
    async def _save_cached_index(self) -> bool:
        """Save FAISS index and chunks to cache."""
        if self.index is None or not self.chunks:
            return False
        
        try:
            index_path = self.cache_dir / "faiss_index.bin"
            chunks_path = self.cache_dir / "chunks.pkl"
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save chunks
            with open(chunks_path, "wb") as f:
                pickle.dump(self.chunks, f)
            
            logger.info("Saved knowledge library index to cache")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cached index: {e}")
            return False
    
    async def refresh_index(self) -> bool:
        """Rebuild the index from scratch."""
        logger.info("Refreshing knowledge library index...")
        self.index = None
        self.chunks = []
        self.initialized = False
        return await self.initialize()
    
    def get_library_stats(self) -> Dict:
        """Get statistics about the knowledge library."""
        return {
            "initialized": self.initialized,
            "num_chunks": len(self.chunks),
            "index_size": self.index.ntotal if self.index else 0,
            "library_path": str(self.library_path),
            "categories": list(set(c.category for c in self.chunks)) if self.chunks else [],
        }


# Singleton instance
_knowledge_library: Optional[KnowledgeLibrary] = None


async def get_knowledge_library() -> KnowledgeLibrary:
    """Get or create knowledge library singleton."""
    global _knowledge_library
    if _knowledge_library is None:
        _knowledge_library = KnowledgeLibrary()
        await _knowledge_library.initialize()
    return _knowledge_library