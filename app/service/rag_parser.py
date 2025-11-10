from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.llm.model import OllamaClient
from app.utils.html import extract_rich_text_from_html_enhanced


# Global embedding model (initialized once)
_embeddings = None


def get_embeddings():
    """Get or initialize the embedding model (singleton pattern)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings


def _try_parse_json(text: str):
    """Extract and parse JSON from LLM output."""
    import re
    candidate = None
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence:
        candidate = fence.group(1)
    else:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        candidate = m.group(1) if m else text.strip()
    try:
        return json.loads(candidate)
    except Exception:
        try:
            import orjson
            return orjson.loads(candidate)
        except Exception:
            return None


def chunk_and_index_html(
    html_content: str, 
    chunk_size: int = 800, 
    overlap: int = 100
) -> Chroma:
    """
    Convert HTML to rich text, chunk it, and create a vector store.
    
    Args:
        html_content: Raw HTML string
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks
    
    Returns:
        Chroma vectorstore with indexed chunks
    """
    # Extract rich text (preserves links, images, data attributes, etc.)
    rich_text = extract_rich_text_from_html_enhanced(html_content)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(rich_text)
    
    # Create documents
    documents = [
        Document(page_content=chunk, metadata={"chunk_id": i, "source": "html"})
        for i, chunk in enumerate(chunks)
    ]
    
    # Create vectorstore with unique collection name (avoid conflicts)
    collection_name = f"html_parse_{uuid.uuid4().hex[:8]}"
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents, 
        embeddings, 
        collection_name=collection_name
    )
    
    return vectorstore


def search_and_retrieve_window(
    vectorstore: Chroma, 
    query: str, 
    offset: int = 0, 
    window_size: int = 20
) -> List[str]:
    """
    Retrieve a WINDOW of chunks starting at offset.
    
    This allows us to explore different parts of the document each iteration
    instead of re-examining the same top-K chunks.
    
    Args:
        vectorstore: Chroma vectorstore
        query: Search query
        offset: Starting position (0-indexed)
        window_size: Number of chunks to retrieve
    
    Returns:
        List of chunk texts
    """
    # Retrieve more than we need, then slice
    total_to_fetch = offset + window_size
    docs = vectorstore.similarity_search(query, k=total_to_fetch)
    
    # Return only the window we want
    window_docs = docs[offset:offset + window_size]
    
    return [doc.page_content for doc in window_docs]


def extract_from_chunks(chunks: List[str], query: str, llm: OllamaClient) -> List[Dict]:
    """
    Extract structured data from chunks using LLM.
    
    Args:
        chunks: List of text chunks
        query: User's extraction query
        llm: Ollama client
    
    Returns:
        List of extracted items (dicts)
    """
    combined_chunks = "\n\n---\n\n".join(chunks)
    
    prompt = f"""Extract structured data from the following HTML content based on the query.
Return ONLY a valid JSON array of objects. Each object should contain the requested fields.


Content:
{combined_chunks}

Query: {query}

JSON Array:"""
    
    output = llm.generate(prompt, max_tokens=7000, temperature=0.0)
    data = _try_parse_json(output)
    
    # Ensure we return a list
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        return []


def deduplicate_results(results: List[Dict]) -> List[Dict]:
    """
    Remove duplicate items from results.
    
    Uses JSON serialization for comparison (order-independent).
    """
    seen = set()
    unique = []
    
    for item in results:
        # Sort keys for consistent comparison
        item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if item_str not in seen:
            seen.add(item_str)
            unique.append(item)
    
    return unique


def parse_with_sliding_window_rag(
    html: str,
    query: str,
    max_iterations: int = 5,
    window_size: int = 30,
    max_total_chunks: int = 150,
    model_id: str | None = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Parse HTML using sliding window RAG approach.
    
    Instead of looking at overlapping top-K chunks each iteration,
    this explores DIFFERENT windows of the document:
      Iter 1: Chunks 0-30
      Iter 2: Chunks 30-60 (completely different!)
      Iter 3: Chunks 60-90 (completely different!)
    
    This finds more items by exploring the entire document space.
    
    Args:
        html: Raw HTML content
        query: Natural language extraction instruction
        max_iterations: Maximum number of windows to examine
        window_size: Chunks per window
        max_total_chunks: Stop after examining this many chunks
        model_id: Ollama model ID (optional)
    
    Returns:
        Tuple of (extracted_data, metadata)
    """
    start = time.time()
    
    # Initialize LLM
    llm = OllamaClient.get(model_id)
    
    # Step 1: Index the HTML
    index_start = time.time()
    vectorstore = chunk_and_index_html(html, chunk_size=800, overlap=100)
    index_time_ms = int((time.time() - index_start) * 1000)
    
    # Count total chunks available
    total_chunks_available = vectorstore._collection.count()
    
    # Step 2: Sliding window retrieval + extraction
    all_results = []
    chunks_examined = 0
    
    # Query variations for different iterations
    query_variations = [
        query,
        f"all {query}",
        f"complete list {query}",
        query.replace("extract", "find all").replace("get", "list all")
    ]
    
    iterations_performed = 0
    
    for iteration in range(max_iterations):
        # Stop if we've examined enough chunks
        if chunks_examined >= max_total_chunks:
            break
        
        # Use different query each iteration
        current_query = query_variations[min(iteration, len(query_variations) - 1)]
        
        # Calculate window offset (sliding window!)
        offset = iteration * window_size
        
        # Stop if offset exceeds available chunks
        if offset >= total_chunks_available:
            break
        
        # Retrieve THIS window (not overlapping with previous!)
        try:
            chunks = search_and_retrieve_window(
                vectorstore, 
                current_query, 
                offset=offset, 
                window_size=window_size
            )
            
            if not chunks:
                break
            
            chunks_examined += len(chunks)
            iterations_performed += 1
            
        except Exception as e:
            break
        
        # Extract from chunks
        results = extract_from_chunks(chunks, query, llm)
        all_results.extend(results)
        
        # Early stopping: if no new results in last 2 iterations
        if iteration > 1:
            unique_so_far = deduplicate_results(all_results)
            unique_before = deduplicate_results(all_results[:-len(results)])
            if len(unique_so_far) == len(unique_before):
                break
    
    # Step 3: Deduplicate final results
    final_results = deduplicate_results(all_results)
    
    elapsed_ms = int((time.time() - start) * 1000)
    
    # Metadata
    meta: Dict[str, Any] = {
        "model": llm.model_id,
        "elapsed_ms": elapsed_ms,
        "index_time_ms": index_time_ms,
        "extraction_time_ms": elapsed_ms - index_time_ms,
        "html_size_chars": len(html),
        "total_chunks_available": total_chunks_available,
        "chunks_examined": chunks_examined,
        "iterations_performed": iterations_performed,
        "window_size": window_size,
        "items_extracted": len(final_results),
        "approach": "sliding_window_rag"
    }
    
    return final_results, meta

