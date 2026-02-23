#!/usr/bin/env python3
"""
Test script for the Knowledge Library RAG System.

This script verifies:
1. Knowledge library initialization
2. Document discovery (finds books in library/)
3. Document indexing and chunking
4. Retrieval of relevant passages
5. Local embeddings functionality
6. Structure-aware chunking results

Usage:
    python scripts/test_knowledge_library.py

Requirements:
    - FAISS installed (pip install faiss-cpu or faiss-gpu)
    - OpenAI API key configured in settings
    - PyMuPDF installed for PDF processing
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.knowledge_library import (
    KnowledgeLibrary,
    DocumentChunk,
    RetrievedPassage,
    get_knowledge_library,
    FAISS_AVAILABLE,
    OPENAI_AVAILABLE,
    PYMUPDF_AVAILABLE,
)
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("test_knowledge_library")


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}> {text}{Colors.RESET}")
    print(f"{Colors.BLUE}{'-'*50}{Colors.RESET}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}[OK] {text}{Colors.RESET}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}[X] {text}{Colors.RESET}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}[!] {text}{Colors.RESET}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.CYAN}[i] {text}{Colors.RESET}")


async def test_dependencies() -> bool:
    """Test that all required dependencies are available."""
    print_section("Checking Dependencies")
    
    all_ok = True
    
    if FAISS_AVAILABLE:
        print_success("FAISS is installed")
    else:
        print_error("FAISS is not installed (pip install faiss-cpu)")
        all_ok = False
    
    if OPENAI_AVAILABLE:
        print_success("OpenAI is installed")
    else:
        print_error("OpenAI is not installed (pip install openai)")
        all_ok = False
    
    if PYMUPDF_AVAILABLE:
        print_success("PyMuPDF is installed")
    else:
        print_warning("PyMuPDF is not installed (pip install pymupdf) - PDF processing unavailable")
    
    return all_ok


async def test_document_discovery(library: KnowledgeLibrary) -> List[tuple]:
    """Test that documents are discovered in the library."""
    print_section("Testing Document Discovery")
    
    documents = await library._scan_documents()
    
    if not documents:
        print_error("No documents found in library")
        print_info(f"Library path: {library.library_path}")
        print_info(f"Categories configured: {[c.get('name') for c in library.config.get('knowledge_library', {}).get('categories', [])]}")
        return []
    
    print_success(f"Found {len(documents)} document(s)")
    
    for doc_path, category, tags in documents:
        print(f"  {Colors.CYAN}*{Colors.RESET} {doc_path.name}")
        print(f"    Category: {Colors.YELLOW}{category}{Colors.RESET}")
        print(f"    Tags: {Colors.YELLOW}{', '.join(tags) if tags else 'none'}{Colors.RESET}")
        print(f"    Path: {doc_path}")
        print()
    
    return documents


async def test_document_processing(library: KnowledgeLibrary, documents: List[tuple]) -> List[DocumentChunk]:
    """Test document processing and chunking."""
    print_section("Testing Document Processing & Chunking")
    
    all_chunks = []
    
    for doc_path, category, tags in documents:
        print(f"\n{Colors.BOLD}Processing: {doc_path.name}{Colors.RESET}")
        
        chunks = await library._process_document(doc_path, category, tags)
        
        if not chunks:
            print_error(f"No chunks generated from {doc_path.name}")
            continue
        
        print_success(f"Generated {len(chunks)} chunks")
        print_info(f"Chunk size: {library.chunk_size} chars")
        print_info(f"Chunk overlap: {library.chunk_overlap} chars")
        
        # Show sample chunks
        print(f"\n{Colors.BOLD}Sample chunks:{Colors.RESET}")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            preview = chunk.text[:200].replace('\n', ' ')
            print(f"\n  {Colors.MAGENTA}Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}:{Colors.RESET}")
            print(f"  {Colors.CYAN}{preview}...{Colors.RESET}")
        
        if len(chunks) > 3:
            print(f"\n  {Colors.YELLOW}... and {len(chunks) - 3} more chunks{Colors.RESET}")
        
        all_chunks.extend(chunks)
    
    return all_chunks


async def test_embeddings(library: KnowledgeLibrary, chunks: List[DocumentChunk]) -> bool:
    """Test embedding generation."""
    print_section("Testing Embedding Generation")
    
    if not chunks:
        print_error("No chunks to generate embeddings for")
        return False
    
    print_info(f"Generating embeddings for {len(chunks)} chunks...")
    print_info(f"Using model: {library.embedding_model}")
    
    try:
        await library._generate_embeddings(chunks)
        
        # Check how many have embeddings
        chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
        
        if not chunks_with_embeddings:
            print_error("No embeddings were generated")
            return False
        
        print_success(f"Generated embeddings for {len(chunks_with_embeddings)}/{len(chunks)} chunks")
        
        if len(chunks_with_embeddings) > 0:
            embedding_dim = len(chunks_with_embeddings[0].embedding)
            print_info(f"Embedding dimension: {embedding_dim}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to generate embeddings: {e}")
        return False


async def test_faiss_index(library: KnowledgeLibrary, chunks: List[DocumentChunk]) -> bool:
    """Test FAISS index building."""
    print_section("Testing FAISS Index Building")
    
    valid_chunks = [c for c in chunks if c.embedding is not None]
    
    if not valid_chunks:
        print_error("No chunks with embeddings to index")
        return False
    
    try:
        await library._build_faiss_index(valid_chunks)
        
        if library.index is None:
            print_error("FAISS index was not created")
            return False
        
        print_success(f"FAISS index built successfully")
        print_info(f"Index size: {library.index.ntotal} vectors")
        print_info(f"Index dimension: {library.index.d}")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to build FAISS index: {e}")
        return False


async def test_retrieval(library: KnowledgeLibrary) -> List[RetrievedPassage]:
    """Test passage retrieval."""
    print_section("Testing Passage Retrieval")
    
    # Test queries relevant to the economic theory book
    test_queries = [
        "What is the role of government in the economy?",
        "How do markets coordinate resources?",
        "What causes economic booms and busts?",
        "Explain the concept of private property",
        "What is the function of money?",
    ]
    
    all_passages = []
    
    for query in test_queries:
        print(f"\n{Colors.BOLD}Query: {Colors.YELLOW}{query}{Colors.RESET}")
        
        passages = await library.retrieve_relevant_passages(query, top_k=3)
        
        if not passages:
            print_warning("No relevant passages found")
            continue
        
        print_success(f"Retrieved {len(passages)} passage(s)")
        
        for i, passage in enumerate(passages):
            print(f"\n  {Colors.MAGENTA}Result {i+1}:{Colors.RESET}")
            print(f"  {Colors.CYAN}Source:{Colors.RESET} {passage.source}")
            print(f"  {Colors.CYAN}Category:{Colors.RESET} {passage.category}")
            print(f"  {Colors.CYAN}Similarity:{Colors.RESET} {passage.similarity_score:.4f}")
            preview = passage.text[:300].replace('\n', ' ')
            print(f"  {Colors.CYAN}Text:{Colors.RESET} {preview}...")
        
        all_passages.extend(passages)
    
    return all_passages


async def test_library_stats(library: KnowledgeLibrary) -> None:
    """Test and display library statistics."""
    print_section("Library Statistics")
    
    stats = library.get_library_stats()
    
    print(f"{Colors.BOLD}Library Status:{Colors.RESET}")
    print(f"  Initialized: {Colors.GREEN if stats['initialized'] else Colors.RED}{stats['initialized']}{Colors.RESET}")
    print(f"  Number of chunks: {Colors.CYAN}{stats['num_chunks']}{Colors.RESET}")
    print(f"  Index size: {Colors.CYAN}{stats['index_size']}{Colors.RESET}")
    print(f"  Library path: {Colors.CYAN}{stats['library_path']}{Colors.RESET}")
    
    if stats['categories']:
        print(f"  Categories: {Colors.YELLOW}{', '.join(stats['categories'])}{Colors.RESET}")


async def test_caching(library: KnowledgeLibrary) -> bool:
    """Test index caching functionality."""
    print_section("Testing Index Caching")
    
    try:
        # Save cache
        saved = await library._save_cached_index()
        if saved:
            print_success("Index saved to cache")
        else:
            print_warning("Could not save cache (index may be empty)")
        
        # Check cache files
        cache_dir = library.cache_dir
        index_path = cache_dir / "faiss_index.bin"
        chunks_path = cache_dir / "chunks.pkl"
        
        if index_path.exists():
            size_mb = index_path.stat().st_size / (1024 * 1024)
            print_info(f"Cache file: {index_path} ({size_mb:.2f} MB)")
        
        if chunks_path.exists():
            size_kb = chunks_path.stat().st_size / 1024
            print_info(f"Chunks file: {chunks_path} ({size_kb:.2f} KB)")
        
        return saved
        
    except Exception as e:
        print_error(f"Cache test failed: {e}")
        return False


async def run_all_tests() -> Dict[str, Any]:
    """Run all knowledge library tests."""
    print_header("KNOWLEDGE LIBRARY TEST SUITE")
    
    results = {
        "dependencies_ok": False,
        "documents_found": 0,
        "chunks_generated": 0,
        "embeddings_ok": False,
        "index_built": False,
        "retrieval_works": False,
        "cache_works": False,
    }
    
    # Test 1: Dependencies
    results["dependencies_ok"] = await test_dependencies()
    
    if not results["dependencies_ok"]:
        print_error("\nRequired dependencies missing. Cannot continue tests.")
        return results
    
    # Initialize library
    print_section("Initializing Knowledge Library")
    library = KnowledgeLibrary()
    
    initialized = await library.initialize()
    
    if initialized:
        print_success("Knowledge library initialized successfully")
    else:
        print_warning("Library not initialized from cache, will build from scratch")
    
    # Test 2: Document Discovery
    documents = await test_document_discovery(library)
    results["documents_found"] = len(documents)
    
    if not documents:
        print_error("\nNo documents found. Cannot continue tests.")
        return results
    
    # Test 3: Document Processing & Chunking
    chunks = await test_document_processing(library, documents)
    results["chunks_generated"] = len(chunks)
    
    if not chunks:
        print_error("\nNo chunks generated. Cannot continue tests.")
        return results
    
    # Test 4: Embeddings
    results["embeddings_ok"] = await test_embeddings(library, chunks)
    
    if not results["embeddings_ok"]:
        print_error("\nEmbedding generation failed. Cannot continue tests.")
        return results
    
    # Test 5: FAISS Index
    results["index_built"] = await test_faiss_index(library, chunks)
    
    if not results["index_built"]:
        print_error("\nIndex building failed. Cannot continue tests.")
        return results
    
    # Test 6: Retrieval
    passages = await test_retrieval(library)
    results["retrieval_works"] = len(passages) > 0
    
    # Test 7: Statistics
    await test_library_stats(library)
    
    # Test 8: Caching
    results["cache_works"] = await test_caching(library)
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print test summary."""
    print_header("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"{Colors.BOLD}Results:{Colors.RESET}\n")
    
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {test_name.replace('_', ' ').title():.<40} {status}")
    
    print(f"\n{Colors.BOLD}Overall: {passed_tests}/{total_tests} tests passed{Colors.RESET}")
    
    if passed_tests == total_tests:
        print(f"\n{Colors.GREEN}{Colors.BOLD}[OK] All tests passed! Knowledge library is working correctly.{Colors.RESET}")
    elif passed_tests >= total_tests / 2:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}[!] Some tests failed. Knowledge library may have limited functionality.{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}[X] Most tests failed. Knowledge library is not working correctly.{Colors.RESET}")


async def main():
    """Main entry point."""
    try:
        results = await run_all_tests()
        print_summary(results)
        
        # Exit with appropriate code
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        if passed == total:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
