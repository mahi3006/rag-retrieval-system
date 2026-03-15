"""
retrieval.py
------------
Handles querying ChromaDB collections and returning ranked chunks.

Responsibilities:
  - Embed a query and fetch top-k chunks from a collection
  - Track retrieval latency per query
  - Provide a unified retrieve() interface so generation.py and
    eval harness don't care which collection or model is underneath

Hybrid retrieval (BM25 + vector) lives in bonuses/hybrid_retrieval.py
and plugs into the same interface.
"""

import time
from typing import List, Dict, Any

import chromadb

from vectorstore import similarity_search


# ── Core Retrieval ─────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    collection: chromadb.Collection,
    model_wrapper: Dict[str, Any],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieves the top-k most relevant chunks for a query.

    Args:
        query        : The user's question string.
        collection   : A loaded ChromaDB collection (from vectorstore.py).
        model_wrapper: The embedding model wrapper matching the collection.
        k            : Number of chunks to retrieve.

    Returns:
        {
            "query"          : original query string,
            "chunks"         : list of retrieved chunk dicts (text, metadata, distance, rank),
            "latency_ms"     : retrieval time in milliseconds,
            "collection_name": name of the collection queried,
            "model_name"     : name of the embedding model used,
            "k"              : number of results requested,
        }
    """
    start = time.time()

    chunks = similarity_search(
        query=query,
        collection=collection,
        model_wrapper=model_wrapper,
        k=k,
    )

    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "query"          : query,
        "chunks"         : chunks,
        "latency_ms"     : latency_ms,
        "collection_name": collection.name,
        "model_name"     : model_wrapper["name"],
        "k"              : k,
    }


# ── Batch Retrieval ────────────────────────────────────────────────────────────

def retrieve_batch(
    queries: List[str],
    collection: chromadb.Collection,
    model_wrapper: Dict[str, Any],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Runs retrieve() for a list of queries.
    Used by the evaluation harness to process all test queries at once.

    Args:
        queries      : List of question strings.
        collection   : A loaded ChromaDB collection.
        model_wrapper: The matching embedding model wrapper.
        k            : Number of chunks per query.

    Returns:
        List of retrieval result dicts, one per query.
    """
    results = []
    for query in queries:
        result = retrieve(query, collection, model_wrapper, k=k)
        results.append(result)
    return results


# ── Format Retrieved Chunks for Display ───────────────────────────────────────

def format_chunks_for_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into a single context string for the LLM prompt.

    Each chunk is labelled with its rank, source file, and page number
    so the LLM can attribute its answer and we can trace hallucinations.

    Example output:
        [1] Source: mckinsey_fmcg.pdf, Page 4
        AI-driven demand forecasting reduces inventory costs by ...

        [2] Source: arxiv_2401.pdf, Page 12
        Supply chain automation using LLMs has been shown to ...

    Args:
        chunks: List of chunk dicts from retrieve().

    Returns:
        A formatted multi-line string ready to insert into an LLM prompt.
    """
    lines = []
    for chunk in chunks:
        source = chunk["metadata"].get("source", "unknown")
        page   = chunk["metadata"].get("page",   "?")
        lines.append(f"[{chunk['rank']}] Source: {source}, Page {page}")
        lines.append(chunk["text"])
        lines.append("")   # blank line between chunks
    return "\n".join(lines)


def format_sources(chunks: List[Dict[str, Any]]) -> str:
    """
    Returns a compact source citation string for the final answer.

    Example: "Sources: mckinsey_fmcg.pdf (p.4), arxiv_2401.pdf (p.12)"
    """
    seen    = set()
    sources = []
    for chunk in chunks:
        source = chunk["metadata"].get("source", "unknown")
        page   = chunk["metadata"].get("page",   "?")
        key    = f"{source} (p.{page})"
        if key not in seen:
            seen.add(key)
            sources.append(key)
    return "Sources: " + ", ".join(sources)


# ── Retrieval Stats ────────────────────────────────────────────────────────────

def print_retrieval_result(result: Dict[str, Any]) -> None:
    """
    Pretty-prints a retrieval result — useful for debugging and demos.
    """
    print(f"\n[Retrieval Result]")
    print(f"  Query      : {result['query']}")
    print(f"  Collection : {result['collection_name']}")
    print(f"  Model      : {result['model_name']}")
    print(f"  Latency    : {result['latency_ms']}ms")
    print(f"  Top-{result['k']} chunks:")
    for chunk in result["chunks"]:
        source = chunk["metadata"].get("source", "unknown")
        page   = chunk["metadata"].get("page",   "?")
        dist   = round(chunk["distance"], 4)
        print(f"    [{chunk['rank']}] dist={dist} | {source} p.{page}")
        print(f"        {chunk['text'][:120].strip()}...")


# ── Entry Point (for testing) ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from embedding import load_both_models
    from vectorstore import load_collection

    print("[Test] Loading models and collections...")
    models = load_both_models(device="cuda")

    test_queries = [
        "What are the main applications of AI in FMCG supply chains?",
        "How does demand forecasting work with machine learning?",
        "What are the risks of using generative AI in logistics?",
    ]

    # Test all 4 combinations
    combos = [
        ("fixed_minilm",     models["minilm"]),
        ("recursive_bge",    models["bge"]),
    ]

    for col_name, model_wrapper in combos:
        col = load_collection(col_name)
        print(f"\n{'='*60}")
        print(f"Collection: {col_name}")
        print(f"{'='*60}")

        for query in test_queries[:1]:   # just one query per combo in test
            result = retrieve(query, col, model_wrapper, k=3)
            print_retrieval_result(result)

            print("\n  Formatted context (first 500 chars):")
            ctx = format_chunks_for_context(result["chunks"])
            print("  " + ctx[:500].replace("\n", "\n  "))