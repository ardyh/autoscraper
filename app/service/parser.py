from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Tuple

from app.llm.model import OllamaClient
from app.service.rag_parser import parse_with_sliding_window_rag


def _try_parse_json(text: str):
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


def parse_with_llm(html: str, query: str, model_id: str | None = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Parse HTML using RAG (Retrieval Augmented Generation) with sliding window approach.
    
    This is the default method that uses semantic search + iterative extraction
    to find ALL relevant data in the HTML.
    
    Set USE_SIMPLE_PARSER=1 environment variable to use the old direct LLM approach.
    """
    # Check if user wants to use simple parser (for backward compatibility)
    use_simple = os.getenv("USE_SIMPLE_PARSER", "0") == "1"
    
    if use_simple:
        return _parse_with_llm_simple(html, query, model_id)
    
    # Default: Use RAG with sliding window (better for complex/list extraction)
    return parse_with_sliding_window_rag(
        html=html,
        query=query,
        max_iterations=5,
        window_size=15,
        max_total_chunks=150,
        model_id=model_id
    )


def _parse_with_llm_simple(html: str, query: str, model_id: str | None = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Simple direct LLM parsing (no RAG).
    
    This is the old approach: feed entire HTML directly to LLM.
    Works well for small HTML or simple queries, but may miss items in large documents.
    """
    start = time.time()
    # Keep it simple: feed raw HTML and the query only
    prompt = f"HTML:\n{html}\n\nQuery:\n{query}\n"

    client = OllamaClient.get(model_id)
    output = client.generate(prompt, max_tokens=1024, temperature=0.0)
    data = _try_parse_json(output)

    elapsed_ms = int((time.time() - start) * 1000)
    meta: Dict[str, Any] = {
        "model": client.model_id,
        "elapsed_ms": elapsed_ms,
        "prompt_chars": len(prompt),
        "output_preview": (output[:200] + "…") if len(output) > 200 else output,
        "approach": "simple_direct_llm"
    }

    return data, meta


