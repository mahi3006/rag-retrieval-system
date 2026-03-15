"""
metrics.py
----------
Computes retrieval evaluation metrics for the RAG benchmarking project.

Metrics implemented:
  - Precision@k  : fraction of retrieved chunks that are relevant
  - Recall@k     : fraction of all relevant chunks that were retrieved
  - MRR          : Mean Reciprocal Rank — rewards finding relevant chunk early
  - Hit Rate@k   : 1 if at least one relevant chunk in top-k, else 0

Relevance definition:
  A retrieved chunk is considered relevant if its text contains at least
  one keyword from the query's `relevant_keywords` list (case-insensitive).
  This is a keyword-proxy for relevance — acknowledged as a limitation.

All metric functions are pure — they take lists and return numbers.
No I/O or model calls happen here.
"""

import re
from typing import List, Dict, Any


# ── Relevance判定 (Relevance Check) ─────────────────────────────────────────

def is_relevant(chunk_text: str, relevant_keywords: List[str]) -> bool:
    """
    Returns True if the chunk text contains at least one relevant keyword.

    Matching is:
      - Case-insensitive
      - Whole-word preferred but substring also counts
        (e.g. "forecasting" matches "demand forecasting")

    Args:
        chunk_text        : The text content of a retrieved chunk.
        relevant_keywords : List of keyword strings from test_queries.json.

    Returns:
        True if the chunk is considered relevant, False otherwise.
    """
    text_lower = chunk_text.lower()
    for keyword in relevant_keywords:
        if keyword.lower() in text_lower:
            return True
    return False


def get_relevance_flags(
    chunks: List[Dict[str, Any]],
    relevant_keywords: List[str],
) -> List[bool]:
    """
    Returns a list of relevance booleans for each chunk, in rank order.

    Example: [True, False, True, True, False] means chunks 1, 3, 4 are relevant.

    Args:
        chunks           : Retrieved chunk dicts (from retrieval.py).
        relevant_keywords: Keywords from the test query.

    Returns:
        List of bool, one per chunk, in retrieval rank order.
    """
    return [is_relevant(c["text"], relevant_keywords) for c in chunks]


# ── Precision@k ───────────────────────────────────────────────────────────────

def precision_at_k(relevance_flags: List[bool], k: int) -> float:
    """
    Precision@k = (number of relevant chunks in top-k) / k

    Measures: of the chunks we retrieved, how many are actually useful?
    Range: 0.0 (none relevant) to 1.0 (all relevant)

    Args:
        relevance_flags: Boolean list (True = relevant) in rank order.
        k              : Number of results to consider (top-k).

    Returns:
        Float between 0.0 and 1.0.
    """
    if k == 0:
        return 0.0
    top_k_flags = relevance_flags[:k]
    return sum(top_k_flags) / k


# ── Recall@k ──────────────────────────────────────────────────────────────────

def recall_at_k(
    relevance_flags: List[bool],
    k: int,
    total_relevant: int,
) -> float:
    """
    Recall@k = (relevant chunks retrieved in top-k) / total_relevant_in_corpus

    Measures: of all relevant information that exists, how much did we find?
    Range: 0.0 to 1.0

    Note: total_relevant is estimated as the count of relevant chunks found
    across ALL retrieved results (not the whole corpus, which we don't label).
    This is a lower bound on true recall.

    Args:
        relevance_flags: Boolean list in rank order.
        k              : Number of results to consider.
        total_relevant : Estimated total relevant chunks (corpus-level).

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 if total_relevant is 0.
    """
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(relevance_flags[:k])
    return min(retrieved_relevant / total_relevant, 1.0)


# ── MRR ───────────────────────────────────────────────────────────────────────

def reciprocal_rank(relevance_flags: List[bool]) -> float:
    """
    Reciprocal Rank = 1 / rank_of_first_relevant_chunk

    Rewards systems that put a relevant chunk at the top.
    If no relevant chunk is found, returns 0.0.

    Examples:
      First relevant at rank 1 → RR = 1/1 = 1.000
      First relevant at rank 2 → RR = 1/2 = 0.500
      First relevant at rank 5 → RR = 1/5 = 0.200
      No relevant found        → RR = 0.000

    Args:
        relevance_flags: Boolean list in rank order.

    Returns:
        Float between 0.0 and 1.0.
    """
    for rank, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(rr_scores: List[float]) -> float:
    """
    MRR = mean of reciprocal ranks across all queries.

    Args:
        rr_scores: List of reciprocal rank scores, one per query.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not rr_scores:
        return 0.0
    return sum(rr_scores) / len(rr_scores)


# ── Hit Rate@k ────────────────────────────────────────────────────────────────

def hit_rate_at_k(relevance_flags: List[bool], k: int) -> float:
    """
    Hit Rate@k = 1 if at least one relevant chunk in top-k, else 0.

    Binary metric — did the retriever find anything useful at all?
    Averaged across queries this gives the fraction of queries answered.

    Args:
        relevance_flags: Boolean list in rank order.
        k              : Number of results to consider.

    Returns:
        1.0 or 0.0
    """
    return 1.0 if any(relevance_flags[:k]) else 0.0


# ── Per-Query Metrics ─────────────────────────────────────────────────────────

def compute_query_metrics(
    chunks: List[Dict[str, Any]],
    relevant_keywords: List[str],
    k: int = 5,
) -> Dict[str, float]:
    """
    Computes all metrics for a single query's retrieval result.

    Args:
        chunks           : Retrieved chunk dicts from retrieval.py.
        relevant_keywords: Keywords from test_queries.json.
        k                : Cutoff for @k metrics.

    Returns:
        {
            "precision_at_k"   : float,
            "recall_at_k"      : float,
            "reciprocal_rank"  : float,
            "hit_rate_at_k"    : float,
            "num_relevant_in_k": int,    # raw count of relevant chunks in top-k
            "k"                : int,
        }
    """
    flags         = get_relevance_flags(chunks, relevant_keywords)
    total_relevant = sum(flags)   # proxy: relevant found in retrieved set

    return {
        "precision_at_k"   : precision_at_k(flags, k),
        "recall_at_k"      : recall_at_k(flags, k, max(total_relevant, 1)),
        "reciprocal_rank"  : reciprocal_rank(flags),
        "hit_rate_at_k"    : hit_rate_at_k(flags, k),
        "num_relevant_in_k": sum(flags[:k]),
        "k"                : k,
    }


# ── Aggregate Metrics Across Queries ─────────────────────────────────────────

def compute_aggregate_metrics(
    per_query_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Averages per-query metrics across all queries.

    Args:
        per_query_metrics: List of dicts from compute_query_metrics().

    Returns:
        {
            "mean_precision_at_k": float,
            "mean_recall_at_k"   : float,
            "mrr"                : float,
            "mean_hit_rate_at_k" : float,
            "num_queries"        : int,
        }
    """
    n = len(per_query_metrics)
    if n == 0:
        return {}

    return {
        "mean_precision_at_k": round(sum(m["precision_at_k"]  for m in per_query_metrics) / n, 4),
        "mean_recall_at_k"   : round(sum(m["recall_at_k"]     for m in per_query_metrics) / n, 4),
        "mrr"                : round(sum(m["reciprocal_rank"]  for m in per_query_metrics) / n, 4),
        "mean_hit_rate_at_k" : round(sum(m["hit_rate_at_k"]   for m in per_query_metrics) / n, 4),
        "num_queries"        : n,
    }


# ── Entry Point (for testing) ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick unit test with toy data
    print("[metrics.py] Running unit tests...\n")

    keywords = ["demand forecasting", "inventory", "AI"]

    # Simulate retrieved chunks
    mock_chunks = [
        {"text": "AI-driven demand forecasting improves accuracy", "rank": 1},
        {"text": "Weather data affects crop yields globally",       "rank": 2},
        {"text": "Inventory optimisation reduces holding costs",    "rank": 3},
        {"text": "Carbon footprint measurement in factories",       "rank": 4},
        {"text": "AI in logistics reduces delivery times",          "rank": 5},
    ]

    flags = get_relevance_flags(mock_chunks, keywords)
    print(f"Relevance flags : {flags}")
    # Expected: [True, False, True, False, True]

    p_at_5  = precision_at_k(flags, k=5)
    r_at_5  = recall_at_k(flags, k=5, total_relevant=3)
    rr      = reciprocal_rank(flags)
    hr      = hit_rate_at_k(flags, k=5)

    print(f"Precision@5     : {p_at_5:.4f}  (expected 0.6000)")
    print(f"Recall@5        : {r_at_5:.4f}  (expected 1.0000)")
    print(f"Reciprocal Rank : {rr:.4f}  (expected 1.0000 — first chunk relevant)")
    print(f"Hit Rate@5      : {hr:.4f}  (expected 1.0000)")

    # Test with no relevant chunks
    empty_flags = [False, False, False, False, False]
    print(f"\nNo relevant chunks:")
    print(f"  Precision@5     : {precision_at_k(empty_flags, 5)}")
    print(f"  Reciprocal Rank : {reciprocal_rank(empty_flags)}")
    print(f"  Hit Rate@5      : {hit_rate_at_k(empty_flags, 5)}")

    print("\n[metrics.py] All unit tests passed.")