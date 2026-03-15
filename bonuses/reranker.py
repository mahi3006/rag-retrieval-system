"""
bonuses/reranker.py
-------------------
Bonus 2: Cross-encoder reranking of retrieved chunks.

WHY RERANKING:
  Vector search retrieves chunks based on embedding similarity — a fast
  but approximate measure of relevance. A cross-encoder reranker reads
  the (query, chunk) pair together and produces a much more accurate
  relevance score, at the cost of being slower.

  Two-stage pipeline:
    Stage 1 — Vector search retrieves top-20 candidates (fast, approximate)
    Stage 2 — Cross-encoder reranks top-20 → returns best 5 (slow, accurate)

  This gives the speed of vector search with near-gold-standard precision.

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
  Trained on MS MARCO passage retrieval dataset.
  Takes (query, passage) as input, outputs a relevance score.
  Free, runs locally, ~80MB.

USAGE:
  from bonuses.reranker import Reranker
  reranker = Reranker()
  results  = reranker.rerank(query, initial_chunks, top_n=5)
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Configuration ──────────────────────────────────────────────────────────────

RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_FETCH_K = 20    # fetch this many from vector search before reranking
DEFAULT_TOP_N   = 5     # return this many after reranking


# ── Reranker ───────────────────────────────────────────────────────────────────

class Reranker:
    """
    Wraps a cross-encoder model for reranking retrieved chunks.

    The cross-encoder reads the full (query, chunk_text) pair jointly —
    unlike bi-encoders (used in vector search) which encode them separately.
    This joint encoding produces much more accurate relevance scores.
    """

    def __init__(self, model_name: str = RERANKER_MODEL, device: str = "cuda"):
        """
        Loads the cross-encoder model.

        Args:
            model_name: HuggingFace model name.
            device    : "cuda" or "cpu".
        """
        print(f"[Reranker] Loading {model_name} on {device}...")
        start = time.time()

        self.model      = CrossEncoder(model_name, device=device)
        self.model_name = model_name

        print(f"[Reranker] Loaded in {time.time() - start:.1f}s")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = DEFAULT_TOP_N,
    ) -> Dict[str, Any]:
        """
        Reranks a list of retrieved chunks by cross-encoder relevance score.

        Args:
            query  : The user's question string.
            chunks : Initial retrieved chunks (from vector/hybrid search).
            top_n  : How many to return after reranking.

        Returns:
            {
                "query"          : str,
                "chunks"         : top_n reranked chunks (with rerank_score added),
                "latency_ms"     : reranking time in ms,
                "model_name"     : reranker model name,
                "initial_k"      : number of chunks before reranking,
                "top_n"          : number of chunks after reranking,
            }
        """
        if not chunks:
            return {
                "query"     : query,
                "chunks"    : [],
                "latency_ms": 0.0,
                "model_name": self.model_name,
                "initial_k" : 0,
                "top_n"     : top_n,
            }

        start = time.time()

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs — cross-encoder returns a relevance logit per pair
        scores = self.model.predict(pairs)

        # Attach scores to chunks and sort descending
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            enriched = dict(chunk)
            enriched["rerank_score"]    = float(score)
            enriched["original_rank"]   = chunk.get("rank", -1)
            scored_chunks.append(enriched)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Re-assign ranks after reranking
        for i, chunk in enumerate(scored_chunks):
            chunk["rank"] = i + 1

        top_chunks = scored_chunks[:top_n]
        latency_ms = round((time.time() - start) * 1000, 2)

        return {
            "query"     : query,
            "chunks"    : top_chunks,
            "latency_ms": latency_ms,
            "model_name": self.model_name,
            "initial_k" : len(chunks),
            "top_n"     : top_n,
        }


# ── Full Pipeline: Retrieve → Rerank ──────────────────────────────────────────

def retrieve_and_rerank(
    query: str,
    collection,
    model_wrapper: Dict[str, Any],
    reranker: "Reranker",
    initial_k: int = INITIAL_FETCH_K,
    final_k: int   = DEFAULT_TOP_N,
) -> Dict[str, Any]:
    """
    Full two-stage pipeline: vector retrieval → cross-encoder reranking.

    Stage 1: Fetch initial_k candidates with vector search (fast)
    Stage 2: Rerank to final_k with cross-encoder (accurate)

    Args:
        query        : User's question.
        collection   : ChromaDB collection.
        model_wrapper: Embedding model wrapper.
        reranker     : Loaded Reranker instance.
        initial_k    : Candidates to fetch in stage 1 (default 20).
        final_k      : Final results after reranking (default 5).

    Returns:
        Combined result dict with retrieval + reranking latencies.
    """
    from retrieval import retrieve

    # Stage 1: vector retrieval
    retrieval_result = retrieve(
        query=query,
        collection=collection,
        model_wrapper=model_wrapper,
        k=initial_k,
    )

    # Stage 2: reranking
    rerank_result = reranker.rerank(
        query=query,
        chunks=retrieval_result["chunks"],
        top_n=final_k,
    )

    return {
        "query"                   : query,
        "chunks"                  : rerank_result["chunks"],
        "retrieval_latency_ms"    : retrieval_result["latency_ms"],
        "rerank_latency_ms"       : rerank_result["latency_ms"],
        "total_latency_ms"        : retrieval_result["latency_ms"] + rerank_result["latency_ms"],
        "collection_name"         : retrieval_result["collection_name"] + "_reranked",
        "embedding_model"         : model_wrapper["name"],
        "reranker_model"          : reranker.model_name,
        "initial_k"               : initial_k,
        "final_k"                 : final_k,
    }


# ── Compare: With vs Without Reranking ────────────────────────────────────────

def compare_with_without_reranking(
    query: str,
    collection,
    model_wrapper: Dict[str, Any],
    reranker: "Reranker",
    k: int = 5,
) -> None:
    """
    Prints a side-by-side comparison of results before and after reranking.
    Shows rank changes clearly.
    """
    from retrieval import retrieve

    # Without reranking (top-5 directly)
    base_result = retrieve(query, collection, model_wrapper, k=k)

    # With reranking (fetch 20, rerank to 5)
    reranked_result = retrieve_and_rerank(
        query=query,
        collection=collection,
        model_wrapper=model_wrapper,
        reranker=reranker,
        initial_k=20,
        final_k=k,
    )

    print(f"\n{'='*65}")
    print(f"QUERY: {query}")
    print(f"{'='*65}")

    print(f"\n── Without Reranking (vector top-{k}) ──")
    for chunk in base_result["chunks"]:
        src  = chunk["metadata"].get("source", "?")
        page = chunk["metadata"].get("page",   "?")
        dist = round(chunk["distance"], 4)
        print(f"  [{chunk['rank']}] {src} p.{page} | dist={dist}")
        print(f"      {chunk['text'][:120].strip()}...")

    print(f"\n── With Reranking (vector 20 → rerank → top-{k}) ──")
    for chunk in reranked_result["chunks"]:
        src      = chunk["metadata"].get("source", "?")
        page     = chunk["metadata"].get("page",   "?")
        score    = round(chunk.get("rerank_score", 0), 3)
        orig_r   = chunk.get("original_rank", "?")
        print(f"  [{chunk['rank']}] {src} p.{page} | rerank_score={score} (was rank {orig_r})")
        print(f"      {chunk['text'][:120].strip()}...")

    print(f"\n  Base latency    : {base_result['latency_ms']}ms")
    print(f"  Reranked total  : {reranked_result['total_latency_ms']}ms "
          f"(retrieval {reranked_result['retrieval_latency_ms']}ms + "
          f"rerank {reranked_result['rerank_latency_ms']}ms)")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from embedding import load_both_models
    from vectorstore import load_collection

    models     = load_both_models(device="cuda")
    collection = load_collection("recursive_bge")
    reranker   = Reranker(device="cuda")

    test_queries = [
        "What are the main applications of AI in FMCG supply chains?",
        "How do companies handle model drift in deployed supply chain AI systems?",
        "What challenges exist when implementing AI in data-light environments?",
    ]

    for query in test_queries:
        compare_with_without_reranking(
            query=query,
            collection=collection,
            model_wrapper=models["bge"],
            reranker=reranker,
            k=5,
        )